"""
V10 Phase 2 with per-epoch weight saving.
Each experiment saves to its own folder: seed_exps/{name}/
"""
import yaml, torch, numpy as np, argparse, os, sys, time, csv, copy, logging
sys.path.insert(0, os.path.dirname(__file__))
from src.loader.data_iterator import DataIterator
from src.trainer.train import Trainer
from src.lib.utils import data_prepare
from src.model.stgcn_n.stgcn import STGCN_n

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class V10SeedTrainer(Trainer):
    def __init__(self, configs, model, train_iterator, val_iterator, test_iterator,
                 vae_source_state, lambda_l2=1.0, exp_dir='./seed_exps/default'):
        super().__init__(configs, model, train_iterator, val_iterator, test_iterator)
        self.vae_source_state = {k: v.clone() for k, v in vae_source_state.items()}
        self.lambda_l2 = lambda_l2
        self.exp_dir = exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        self.epoch_results = []

    def _train_epoch(self, epoch):
        self.model.train()
        for name, param in self.model.named_parameters():
            param.requires_grad = True
        epoch_loss = 0
        n_iter = 0
        lam = self.lambda_l2
        for values in self.train_iterator:
            values = values.to(self.device)
            self.opt.zero_grad()
            y_pred, future_data, embedding, vae_loss = self._iter(values=values, epoch=epoch)
            y = values[..., -self.horizon:]
            if self.configs.get('aug_node'):
                if self.aug_node_epoch: y = future_data
                else: y_pred = y_pred[:, :values.shape[1]]
            iter_loss = 0
            if self.aug_node_epoch:
                iter_loss = 0.01 * vae_loss
                for name, param in self.model.named_parameters():
                    if 'vae' not in name: param.requires_grad = False
                y = y[:, values.shape[1]:]
                y_pred = y_pred[:, values.shape[1]:]
                if self.configs.get('fst_loss', False):
                    iter_loss += self.get_mae(y=y[..., 0, :], y_pred=y_pred)
                l2_sum = 0.0
                for name, param in self.model.named_parameters():
                    if 'vae' in name and name in self.vae_source_state:
                        src = self.vae_source_state[name].to(self.device)
                        l2_sum += ((param - src) ** 2).sum()
                iter_loss += lam * l2_sum
                iter_loss.backward(); self.opt.step()
                n_iter += 1; epoch_loss += iter_loss.item()
                continue
            elif self.aug_node_n_epoch:
                for name, param in self.model.named_parameters():
                    if 'vae' in name: param.requires_grad = False
            if not self.configs.get('train_normalized', False):
                y = y * self.scaler[1][..., :] + self.scaler[0][..., :]
                y_pred = y_pred * self.scaler[1][..., 0, :] + self.scaler[0][..., 0, :]
            iter_loss += self.get_mae(y=y[..., 0, :], y_pred=y_pred)
            iter_loss.backward(); self.opt.step()
            n_iter += 1; epoch_loss += iter_loss.item()
        epoch_loss /= max(1, n_iter)
        return epoch_loss

    def train(self):
        for epoch in range(self.configs['epochs']):
            self.aug_node_epoch = self.configs.get('aug_node') and epoch % 2 == 0
            self.aug_node_n_epoch = self.configs.get('aug_node') and (epoch % 2 == 1)
            train_loss = self._train_epoch(epoch)
            if self.configs.get('aug_node') and self.aug_node_epoch: continue
            if self.configs.get('model_name') != 'HI': self.scheduler.step()
            eval_loss = self.eval(epoch)
            test_loss, metrics = self.test()
            result = {
                'epoch': epoch, 'test_loss': test_loss, 'eval_loss': eval_loss,
                'mae_12': metrics[0], 'rmse_12': metrics[1], 'mape_12': metrics[2],
                'mae_6': metrics[3], 'rmse_6': metrics[4], 'mape_6': metrics[5],
                'mae_3': metrics[6], 'rmse_3': metrics[7], 'mape_3': metrics[8],
            }
            self.epoch_results.append(result)
            weight_path = os.path.join(self.exp_dir, 'model_epoch{:03d}_mae12_{:.4f}.pt'.format(epoch, metrics[0]))
            torch.save(self.model.state_dict(), weight_path)
            csv_path = os.path.join(self.exp_dir, 'epoch_results.csv')
            with open(csv_path, 'w') as f:
                w = csv.DictWriter(f, fieldnames=result.keys())
                w.writeheader()
                for r in self.epoch_results: w.writerow(r)
            if eval_loss < self.loss_min:
                self.loss_min = eval_loss; self.best_test = test_loss
                self.best_metrics = metrics; self.n_tolerance = 0
                self.best_model = copy.deepcopy(self.model.state_dict())
            else:
                self.n_tolerance += 1
                if self.n_tolerance >= self.configs.get('tolerance', 20):
                    logger.info('Early stopping at epoch {}'.format(epoch)); break
        logger.info('Training finished')
        self.model.load_state_dict(self.best_model)
        test_loss, metrics = self.test()
        print('FINAL: mae_12={:.4f} mae_6={:.4f} mae_3={:.4f}'.format(metrics[0], metrics[3], metrics[6]))
        best_path = os.path.join(self.exp_dir, 'model_best_mae12_{:.4f}.pt'.format(metrics[0]))
        torch.save(self.best_model, best_path)
        return test_loss

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--vae_source_ckpt', type=str, required=True)
    p.add_argument('--lambda_l2', type=float, default=0.001)
    p.add_argument('--exp_dir', type=str, required=True)
    a = p.parse_args()
    with open(a.config) as f: cfg = yaml.safe_load(f)
    cfg['device'] = a.device
    cfg['save_dir'] = a.exp_dir
    os.makedirs(a.exp_dir, exist_ok=True)
    train_values, val_values, test_values, scaler = data_prepare(cfg)
    train_iter = DataIterator(configs=cfg, values=train_values, scaler=scaler, mode='train')
    val_iter = DataIterator(configs=cfg, values=val_values, scaler=scaler, mode='val')
    test_iter = DataIterator(configs=cfg, values=test_values, scaler=scaler, mode='test')
    src_ckpt = torch.load(a.vae_source_ckpt, map_location='cpu', weights_only=False)
    vae_source_state = {}
    for k, v in src_ckpt.items():
        if 'vae' in k: vae_source_state[k] = v.clone()
    print('Loaded {} VAE params from {}'.format(len(vae_source_state), a.vae_source_ckpt))
    model = STGCN_n(cfg=cfg)
    model_state = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in vae_source_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v; loaded += 1
        elif k in model_state:
            print('  SKIP {}: {} vs {}'.format(k, model_state[k].shape, v.shape))
            skipped += 1
    model.load_state_dict(model_state)
    print('VAE init: {} loaded, {} skipped'.format(loaded, skipped))
    trainer = V10SeedTrainer(
        configs=cfg, model=model,
        train_iterator=train_iter, val_iterator=val_iter, test_iterator=test_iter,
        vae_source_state=vae_source_state, lambda_l2=a.lambda_l2, exp_dir=a.exp_dir
    )
    trainer.train()

if __name__ == '__main__':
    main()
