from pinnPack import network, pinnUtils
import numpy as np
from tqdm import trange
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from imblearn.over_sampling import SMOTE

class phyModel():
    def __init__(self, tx_init, uv_init, tx_col, tx_exp, uv_exp, txU_bound, txL_bound,
                 layers, parameters, optimizer, net = 'DNN',
                 act = 'tanh', encode_dim = None, causal_decay = None, curriculum = None, progressive = None, smote = None,
                 epsilon_weight = 1e-3,  lambda_bound = 1, lambda_init = 1, lambda_r = 1, scheduler = False,):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        #log
        self.thisLog = pinnUtils.DictLogger()
        self.metrics = {'loss': 0,
                       'mse': float('inf')}
        #Model
        self.net = net
        self.layers = layers 
        if self.net == 'DNN':
            self.dnn = network.DNN(layers, act = act, encode_dim = encode_dim)
        else: #Pretrained model path
            self.dnn = network.DNN(layers, act = act, encode_dim = encode_dim)
            state_dict = torch.load(self.net)
            self.dnn.load_state_dict(state_dict)
        self.dnn.to(self.device) 
        
        #Dataset Preparation 
        self.t_init = torch.tensor(tx_init[0], requires_grad = True).float().to(self.device); 
        self.x_init = torch.tensor(tx_init[1], requires_grad = True).float().to(self.device); 
        
        self.tU_bound =  torch.tensor(txU_bound[0], requires_grad = True).float().to(self.device); 
        self.xU_bound =  torch.tensor(txU_bound[1], requires_grad = True).float().to(self.device); 
        self.tL_bound =  torch.tensor(txL_bound[0], requires_grad = True).float().to(self.device); 
        self.xL_bound =  torch.tensor(txL_bound[1], requires_grad = True).float().to(self.device); 
        
        self.t_col = torch.tensor(tx_col[0], requires_grad = True).float().to(self.device); 
        self.x_col = torch.tensor(tx_col[1], requires_grad = True).float().to(self.device);
        
        self.t_exp = torch.tensor(tx_exp[0], requires_grad = True).float().to(self.device); 
        self.x_exp = torch.tensor(tx_exp[1], requires_grad = True).float().to(self.device);
        
        uv_init = tuple(torch.tensor(arr) for arr in uv_init)
        uv_exp = tuple(torch.tensor(arr) for arr in uv_exp)
        self.uv_init = torch.cat(uv_init, axis = 1).float().to(self.device)
        self.uv_exp = torch.cat(uv_exp, axis = 1).float().to(self.device)
        
        #Parameters
            #loss weight parameters
        self.lambda_r = lambda_r
        self.lambda_init = lambda_init
        self.lambda_bound = lambda_bound
            #causal decay
        self.causal_decay = causal_decay
            #curriculum
        self.curriculum = curriculum
            #ff_encod
        self.encode_dim = encode_dim
            #progressive 
        self.progressive = progressive
        self.smote_threshold = 0
        if smote is not None:
            for k, v in smote.items():
                setattr(self, k, v) #smote_threshold, smote_num, smote_epoch
            self.SMOTE = SMOTE() #need improvement
            self.thisLog.add({"sample number": len(self.t_col)})
        for k,v in parameters.items():
            setattr(self, k, v)
        
        #Optimizers and Schedulers
        self.optimizer = pinnUtils.choose_optimizer(optimizer, self.dnn.parameters())
        self.iter = 0 
        self.scheduler = None
        if scheduler:
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.9 ** (self.iter // 5000))
        self.curr_on = None

        if self.causal_decay is not None:
            self.update_causal()
            self.epsilon_weight = epsilon_weight

    def predict(self, t,x):
        return self.dnn((t,x)) 
    
    def compute_grads(self, output, inputs, order = 1):
        grads = output
        for _ in range(order):
            grads = torch.autograd.grad(
                grads, inputs, 
                grad_outputs = torch.ones_like(grads),
                create_graph = True,
                retain_graph = True
            )[0]
        return grads 
    
    def residual_loss(self, t,x): 
        uv = self.predict(t,x)
        
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        
        ut = self.compute_grads(u,t)
        vt = self.compute_grads(v,t) 
        ut2 = self.compute_grads(ut, t)
        vt2 = self.compute_grads(vt, t) 
        ut3 = self.compute_grads(ut2,t)
        vt3 = self.compute_grads(vt2,t) 
        ux = self.compute_grads(u,x) 
        vx = self.compute_grads(v,x)
        
        scalar = (torch.square(u) + torch.square(v))
        residue_u = (ux + self.alpha*u/2
                     -self.beta2/(2)*vt2
                     -self.beta3/(6)*ut3
                     +self.gamma*scalar*v
                    )
        
        residue_v = (vx + self.alpha*v/2
                 +self.beta2/(2)*ut2
                 -self.beta3/(6)*vt3
                 -self.gamma*scalar*u
                )

        residue = torch.cat([residue_u, residue_v], dim = 1)
        return torch.sum(torch.square(residue), dim =1).view(-1, 1)
    
    def boundary_loss(self,tL,xL, tU, xU):
        uvL = self.predict(tL, xL)
        uvU = self.predict(tU, xU)
        uvLt = self.compute_grads(uvL,tL) 
        uvUt = self.compute_grads(uvU, tU) 
        
        drichletUV = F.mse_loss(uvL, uvU)
        neumannUV = F.mse_loss(uvLt, uvUt)
        return drichletUV + neumannUV

    def causal_loss(self,t,x):
        uv_r = self.residual_loss(self.t_col, self.x_col)
        L_t = (self.X_occurs @ uv_r)/self.X_occurs.sum(1).reshape(-1,1)
        W = torch.exp(-self.epsilon_weight*(self.M @ L_t)).detach()
        return W, L_t

    def loss_func(self):
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
        uv_ip = self.predict(self.t_init, self.x_init) 
        loss_b = self.boundary_loss(self.tL_bound, self.xL_bound, 
                                    self.tU_bound, self.xU_bound)

        if not self.causal_decay:
            uv_r = self.residual_loss(self.t_col, self.x_col)
            residue = uv_r.mean()
        else:
            W, L_t = self.causal_loss(self.t_col, self.x_col)
            self.thisLog.add({'causal_minweight': W.min().item()})
            residue = (W*L_t).mean()

        loss_init = F.mse_loss(uv_ip, self.uv_init)
        loss = self.lambda_bound*loss_b + self.lambda_init*loss_init + self.lambda_r*residue

        self.pbar.set_postfix(self.metrics)

        self.thisLog.add({
        "boundary_loss": loss_b.item(),
        "residual_loss": residue.item(),
        "init_loss": loss_init.item(),
        "total_loss": loss.item()
        })
        
        if loss.requires_grad:
            loss.backward()
        
        self.metrics['loss'] = loss.item()
        self.iter += 1
        self.thisLog.add({'epoch': self.iter})

        return loss

    def train(self, epochs = 3000, curriculum: dict = None, progressive: dict = None):
        self.epochs = epochs 
        self.dnn.train()

        if self.curriculum is not None:
            if self.smote_threshold != 0:
                print('Current Curriculum Learning Does Not Support SMOTE Resampling. Proceed into training.')
            self.curriculum_learning(self.curriculum)

            for curr_val in self.curr_range:
                setattr(self, self.curr_on, curr_val)

                if curr_val == self.curr_range[-1] and curriculum['version'] ==2:
                    self.curr_epochs = self.epochs-self.iter
                self.thisLog.add({f'self.curr_on': curr_val})
                self.pbar = trange(self.curr_epochs, desc = f'[CURR] Training on {self.curr_on}: {curr_val:.3}')
                for _ in self.pbar:
                    self.dnn.train() 
                    self.optimizer.step(self.loss_func)
                    if  self.iter % min(self.epochs//10, 100) == 0:
                        self.validate(self.t_exp, self.x_exp, self.uv_exp)

                    if self.scheduler is not None:
                        self.scheduler.step()
                        
        elif self.progressive is not None:
            if self.smote_threshold != 0:
                print('Current Progressive Learning Does Not Support SMOTE Resampling. Proceed into training.')
            self.progressive_learning(self.progressive)
            X_COL = self.x_col.detach().clone()
            T_COL = self.t_col.detach().clone()
            maxL = torch.max(X_COL).cpu().numpy(); minL = torch.min(X_COL).cpu().numpy()
            maxT = torch.max(T_COL).cpu().numpy(); minT = torch.min(T_COL).cpu().numpy()
            N = X_COL.shape[0]

            for index in self.prog_range:
                num_sampling = int(index*N/self.prog_range[-1])
                self.progressive_resampling(np.array([minT, minL]), np.array([maxT, index*maxL/self.prog_range[-1]]), num_sampling)
                assert self.prog_sampling <= self.x_col.shape[0], (
                    f"prog_sampling={self.prog_sampling} exceeds available collocation points ({self.x_col.shape[0]}) "
                )

                if index == self.prog_range[-1] and progressive['version'] ==2:
                    self.prog_epochs = self.epochs-self.iter
                self.thisLog.add({f'self.prog_on': index})
                self.pbar = trange(self.prog_epochs, desc = f'[PROG] Training on Index: {index}')
                for _ in self.pbar:
                    self.dnn.train() 
                    self.optimizer.step(self.loss_func)
                    if  self.iter % min(self.epochs//10, 100) == 0:
                        self.validate(self.t_exp, self.x_exp, self.uv_exp)
                    
                    if self.scheduler is not None:
                        self.scheduler.step()
                
                if not index == self.prog_range[-1]:
                    rand_idx = torch.randint(0, self.x_col.shape[0], (self.prog_sampling,), device=self.x_col.device)
                    x_known = self.x_col[rand_idx]
                    t_known = self.t_col[rand_idx]
                    with torch.no_grad():
                        uv_known = self.predict(t_known, x_known)

                    #Concatenate dataset
                    self.x_init = torch.cat([self.x_init.detach(), x_known.detach()], dim=0).requires_grad_()
                    self.t_init = torch.cat([self.t_init.detach(), t_known.detach()], dim=0).requires_grad_()
                    self.uv_init = torch.cat([self.uv_init.detach(), uv_known.detach()], dim=0).requires_grad_()
            
        else:
            self.pbar = trange(epochs, desc = f'[{self.net}{"/causal" if self.causal_decay else ""}{"/encPBC" if self.encode_dim else ""}] training')
            for _ in self.pbar:
                self.dnn.train()
                self.optimizer.step(self.loss_func)

                if self.iter % 100 == 0:
                    mse = self.validate(self.t_exp, self.x_exp, self.uv_exp)
                if self.scheduler is not None:
                    self.scheduler.step()
                
                if len(self.t_col) > self.smote_max:
                    self.smote_threshold = 0
                if self.smote_threshold !=0 and self.iter%self.smote_epoch == 0:
                    self.smote_resampling(self.t_col, self.x_col)
        return 
    
    def curriculum_learning(self, curriculum):
        print(f"Using curriculum learning on param {curriculum['curr_on']} with target {getattr(self, curriculum['curr_on'])}\n")
        self.curr_on = curriculum['curr_on']
        self.curr_range = np.linspace(*sorted([curriculum['curr_initval'], getattr(self, self.curr_on)]),
                                      curriculum['curr_steps'])
        if curriculum['curr_initval'] > getattr(self, self.curr_on):
            self.curr_range = self.curr_range[::-1] # the bigger the easier
        setattr(self, self.curr_on, self.curr_range[0])
        
        if curriculum['version'] == 1:
            self.curr_epochs = self.epochs // curriculum['curr_steps']
        elif curriculum['version'] == 2:
            assert curriculum['curr_epochs'] * curriculum['curr_steps'] < self.epochs, "More epochs are needed for Curriculum v2"
            self.curr_epochs = curriculum['curr_epochs']
        else:
            raise ValueError(f"Wrong version selection (selected {curriculum['version']} instead of 1 or 2)")
    
    def progressive_learning(self, progressive):
        N = progressive['domain_stages']
        print(f"Progressive PT learning is used for {N} steps")
        self.prog_range = np.arange(1, N+1)
        self.prog_sampling = progressive['sampling']
        
        if progressive['version'] == 1:
            self.prog_epochs = self.epochs//N
        elif progressive['version'] == 2:
            assert progressive['prog_epochs']*progressive['domain_stages'] < self.epochs, "More epochs are needed for Progressive PT v2"
            self.prog_epochs = progressive['prog_epochs']
        else:
            raise ValueError(f"Wrong version selection (selected {progressive['version']} instead of 1 or 2)")
                                                            
                                                            
    def progressive_resampling(self, start, final, n):
        tx_col = pinnUtils.generate_points(start, final, n, seed = None)
        t_col = tx_col[:,0].reshape(-1,1)
        x_col = tx_col[:,1].reshape(-1,1)
        self.t_col = torch.tensor(t_col, requires_grad = True).float().to(self.device); 
        self.x_col = torch.tensor(x_col, requires_grad = True).float().to(self.device);
        if self.causal_decay is not None:
            self.update_causal()

    def smote_resampling(self, t,x):
        tx = torch.cat([t, x], dim = 1)
        if self.causal_decay is not None:
            W, L_t = self.causal_loss(t, x)
            residue = (W*L_t).flatten()
        else:
            residue = (self.residual_loss(t, x)).flatten()
            
        k = int(self.smote_threshold * len(residue))
        __, indices = torch.topk(residue, k=k)
        not_indices = torch.tensor(list(set(range(len(residue)))-set(indices.tolist())), device = indices.device)
        tx1 =tx[indices].detach().cpu().numpy()
        tx0 =tx[not_indices].detach().cpu().numpy()
    
        TX = np.vstack([tx1, tx0])
        Y = np.hstack([np.ones(len(tx1)), np.zeros(len(tx0))])
        TX_resampled, y_resampled = self.SMOTE.fit_resample(TX, Y)
        

        self.t_col = torch.tensor(TX_resampled[:,0].reshape(-1,1), requires_grad = True).float().to(self.device)
        self.x_col = torch.tensor(TX_resampled[:,1].reshape(-1,1), requires_grad = True).float().to(self.device)
        self.optimizer.zero_grad()
        self.thisLog.add({
        "sample number": len(self.t_col),
        })
        if self.causal_decay is not None:
            self.update_causal()
    
    def update_causal(self):
        self.x_unique, unique_indices = torch.unique(self.x_col, return_inverse = True) 
        self.t_unique = self.t_col[:,0][unique_indices]            
        self.X_occurs = torch.stack([x == self.x_unique for x in self.x_col]).float().to(self.device).T
        self.M = torch.triu(torch.ones((self.X_occurs.shape[0], self.X_occurs.shape[0])), diagonal=1).to(self.device).T
            
        
    def validate(self, t,x, uv):
        self.dnn.eval()
        uv_p = self.predict(t,x)
        uv_p = uv_p.detach()
        mse = torch.square(uv_p - uv).mean()
        self.thisLog.add({"valid_mse": mse.cpu().numpy()})
        self.metrics['mse'] = mse.item()
        return mse
    
    def inference(self, tMesh, xMesh):
        T = tMesh.reshape(-1,1); X = xMesh.reshape(-1,1) 
        T = torch.tensor(T).float().to(self.device); X = torch.tensor(X).float().to(self.device)
        self.dnn.eval()
        with torch.no_grad():
            uv = self.predict(T,X)
        
        pulse = torch.complex(uv[:, 0], uv[:, 1]).cpu().numpy()
        pulse = pulse.reshape(tMesh.shape)
        return pulse
    
    def save_model(self, model_path, json_path):
        torch.save(self.dnn.state_dict(), model_path)
        self.thisLog.save(json_path)
    
                                                            



            