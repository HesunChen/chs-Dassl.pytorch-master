import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.modeling import build_head, build_backbone
from dassl.data.transforms import build_transform
from dassl.modeling.ops.utils import create_onehot

INF_STYLES = ["mean", "vote", "most_confident"]

class Encoders(nn .Module):
    def __init__(self, cfg, model_cfg, num_classes, n_source, **kwargs):
        super().__init__()
        self.backbones = nn.ModuleList(
            [build_backbone(
                model_cfg.BACKBONE.NAME,
                verbose=cfg.VERBOSE,
                pretraind=model_cfg.BACKBONE.PRETRAINED,
                **kwargs,
            ) for _ in range(n_source)]
        )
        fdim = self.backbones[0].out_features

        self.heads = None
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN.LAYERS:
            self.heads = nn.ModuleList([
                build_head(
                    model_cfg.HEAD.NAME,
                    verbose=cfg.VERBOSE,
                    in_features=fdim,
                    hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                    activation=model_cfg.HEAD.ACTIVATION,
                    bn=model_cfg.HEAD.BN,
                    dropout=model_cfg.HEAD.DROPOUT,
                    **kwargs,
                ) for _ in range (n_source) ]
            )
            fdim = self.heads[0].out_features

        self._fdim = fdim
        self._n_source = n_source

    @property
    def fdim(self):
            return self._fdim

    def forward(self, i, x):
        f = self.backbones[i](x)
        if self.heads is not None:
            f = self.heads[i](x)

        return f

    def merge_params(self):
        merged_backbone_dict = deepcopy(self.backbones[0].state_dict())
        for i in range(1, self._n_source):
            for var in merged_backbone_dict:
                merged_backbone_dict[var] = merged_backbone_dict[var] + self.backbones[i].state_dict()[var]
        for var in merged_backbone_dict:
            merged_backbone_dict[var] = merged_backbone_dict[var] / self._n_source
        for i in range(self._n_source):
            self.backbones[i].load_state_dict(merged_backbone_dict)    

class Decoders(nn.Module):
    def __init__(self, n_source, fdim, num_classes):
        super().__init__()
        self._n_source = n_source
        self.linears = nn.ModuleList(
            [nn.Linear(fdim, num_classes) for _ in range(n_source*n_source)]
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, i, x):
        x = self.linears[i](x)
        x = self.softmax(x)
        return x
    
    def set_mode(self, i, mode="train"):
        if mode == "train":
            for j in range(self._n_source):
                if j != i:
                    self.linears[i*self._n_source+j].train()
                else:
                    self.linears[i*self._n_source+j].eval()
        elif mode in ["eval", "test"]:
            for j in range(self._n_source):
                if j != i:
                    self.linears[i*self._n_source+j].eval()
                else:
                    self.linears[i*self._n_source+j].eval()
        else:
            raise KeyError

    def merge_params(self):
        for i in range(self._n_source):
            standard_dict = deepcopy(self.linears[self._n_source*i+i].state_dict())
            for j in range(self._n_source):
                if j != i:
                    self.linears[self._n_source*j+i].load_state_dict(standard_dict)

@TRAINER_REGISTRY.register()
class COPADG(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        n_domain = cfg.DATALOADER.TRAIN_X.N_DOMAIN
        batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        if n_domain <= 0:
            n_domain = self.num_source_domains
        self.split_batch = batch_size // n_domain
        self.n_domain = n_domain

        self.conf_thre = cfg.TRAINER.COPA.CONF_THRE

        self.local_iter = 0

        self.inf_style = cfg.TRAINER.COPA.INF_STYLE

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COPA.LOCAL_ITER > 0
        assert cfg.TRAINER.COPA.INF_STYLE in INF_STYLES

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.COPA.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building F")
        self.F = Encoders(cfg, cfg.MODEL, 0, self.num_source_domains)
        self.F.to(self.device)
        print("# params: {:,}".format(count_num_param(self.F)))
        self.optim_F = build_optimizer(self.F, cfg.OPTIM)
        self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
        self.register_model("F", self.F, self.optim_F, self.sched_F)
        fdim = self.F.fdim

        print("Building C")
        self.C = Decoders(self.num_source_domains, fdim, self.num_classes)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)
    
    def forward_backward(self, batch):
        parsed_data = self.parse_batch_train(batch)
        input, input2, label, domain = parsed_data

        input = torch.split(input, self.split_batch, 0)
        input2 = torch.split(input2, self.split_batch, 0)
        label = torch.split(label, self.split_batch, 0)
        domain = torch.split(domain, self.split_batch, 0)
        domain = [d[0].item() for d in domain]

        acc = 0
        loss_x= 0
        loss_cr = 0
        loss = 0
        
        for input_i, input2_i, label_i, i in zip(input, input2, label, domain):
            cr_s = [j for j in domain if j!= i]

            feat_i = self.F(i, input_i)
            feat2_i = self.F(i, input2_i)
            pred_i = self.C(i*self.n_domain+i, feat_i)

            acc += compute_accuracy(pred_i.detach(), label_i.max(1)[1])[0].item()
            
            loss_x += (-label_i * torch.log(pred_i + 1e-5)).sum(1).mean()

            for j in cr_s:
                pred_j = self.C(i*self.n_domain+j, feat2_i)
                loss_cr += (-label_i * torch.log(pred_j + 1e-5)).sum(1).mean()

            loss += loss_x
            loss += loss_cr
                
        loss /= 3
        acc /= 3    
        self.local_iter += 1
        self.model_backward_and_update(loss)
        
        if self.local_iter % self.cfg.TRAINER.COPA.LOCAL_ITER == 0:
            self.F.merge_params()
            self.C.merge_params()
            self.local_iter = 0


        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        loss_summary = {
            "acc": acc,
            "loss_x": loss_x.item(),
            "loss_cr": loss_cr.item()
        }

        return loss_summary

    def set_model_mode(self, mode="train"):
        if mode == "train":
            self.F.train()
            for i in range(self.n_domain): 
                self.C.set_mode(i, mode)

        if mode in ["test", "eval"]:     
            self.F.eval()   
            for i in range(self.n_domain):
                self.C.set_mode(i, mode)

    def parse_batch_train(self, batch):
        input = batch["img"]
        input2 = batch["img2"]
        label = batch["label"]
        domain = batch["domain"]

        label = create_onehot(label, self.num_classes)

        input = input.to(self.device)
        input2 = input2.to(self.device)
        label = label.to(self.device)

        return input, input2, label, domain

    def model_inference(self, input):
        f = self.F(0, input)
        p = []
        for k in range(self.n_domain):
            p_k = self.C(k, f)
            p_k = p_k.unsqueeze(1)
            p.append(p_k)

        if self.inf_style == "mean":
            p = torch.cat(p, 1)
            res = p.mean(1)

        elif self.inf_style == "vote":
            p = torch.cat(p, 1)
            labels = p.max(2)[1]
            one_hots = F.one_hot(labels, self.num_classes)
            votes = one_hots.sum(1).squeeze(1)
            conf_mean = p.mean(1)
            normalized_mean = conf_mean / conf_mean.sum(1).unsqueeze(1)
            res = votes + normalized_mean
            
        elif self.inf_style == "most_confident":
            p = torch.cat(p, 2)
            labels = p.max(2)[1] % self.n_domain
            res = F.one_hot(labels, self.num_classes)
        return res
