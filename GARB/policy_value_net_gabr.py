import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class ResBlock(nn.Module):
    # 初始化策略网络和价值网络结构
    """policy-value network module"""
    def __init__(self, num_filters=128):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_filters,)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=num_filters,out_channels=num_filters,kernel_size=(3,3),stride=(1,1),padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters,)
        self.act2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_filters,out_channels=num_filters,kernel_size=(3,3),stride=(1,1),padding=1)
    def forward(self,x):
        y = self.bn1(x)
        y = self.act1(y)
        y = self.conv1(y)
        y = self.bn2(y)
        y = self.act2(y)
        y = self.conv2(y)
        y = x+y
        return y


class GlobalPoolingLayer(nn.Module):
    def __init__(self, board_width, num_channels=128):
        super().__init__()
        self.num_channels = num_channels
        self.board_width = board_width
    
    def forward(self,G):
        #计算平均值
        mean = torch.mean(G, dim=[2,3], keepdim=True)

        #每个通道的最大值
        max_pool = torch.max(G, dim=2, keepdim=True)[0]
        max_pool = torch.max(max_pool, dim=3, keepdim=True)[0]

        #连接
        global_features = torch.cat((mean, max_pool), dim=1)
        global_pooled = global_features.view(global_features.size(0), self.num_channels * 2)

        #取均值使其形状匹配
        return global_pooled


class GARB(nn.Module):
    # 初始化策略网络和价值网络结构
    """policy-value network module"""
    def __init__(self, board_width, board_height, num_channels=128):
        super().__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.bn1 = nn.BatchNorm2d(num_channels,)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels//3, kernel_size=(3,3), stride=(1,1), padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels,)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels//3*2, kernel_size=(3,3), stride=(1,1), padding=1)
        self.bn3 = nn.BatchNorm2d(num_channels//3,)
        self.relu3 = nn.ReLU()
        self.global_pooling = GlobalPoolingLayer(self.board_width, num_channels//3)
        self.fc = nn.Linear(num_channels//3*2, num_channels//3*2)
        self.bn4 = nn.BatchNorm2d(num_channels//3*2,)
        self.relu4 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=num_channels//3*2, out_channels=num_channels, kernel_size=(3,3), stride=(1,1), padding=1)

    def forward(self, x):
        y = self.conv1(self.relu1(self.bn1(x)))
        z = self.conv2(self.relu2(self.bn2(x)))
        y = self.relu3(self.bn3(y))
        y = self.global_pooling(y)
        y = self.fc(y)
        y = y.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.board_width, self.board_height)
        z = z + y
        z = self.conv3(self.relu4(self.bn4(z)))
        z = z + x

        return z


class Net(nn.Module):
    # 初始化策略网络和价值网络结构
    """policy-value network module"""
    def __init__(self, board_width, board_height, num_res_blocks=5):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Residual Blocks + GARBs
        # 这里的num_res_blocks必须是奇数
        self.res_blocks = nn.ModuleList([ResBlock() if _ % 2 == 0 else GARB(self.board_width, self.board_height) for _ in range(num_res_blocks)])

        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height,
                                 board_width*board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    # 对策略和价值网络分别forward
    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        for layer in self.res_blocks:
            x = layer(x)
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act))
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val    



class PolicyValueNet():
    """policy-value network """
    # 初始化网络参数和优化器
    # 如果是已经训练完模型，进行测试，则需要load
    def __init__(self, board_width, board_height, res_block_num, 
                 model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height, num_res_blocks=res_block_num).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height, num_res_blocks=res_block_num)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)
            print("model loaded", flush=True)

    # 从网络中得到batch对应的概率分布p和v
    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(np.array(state_batch)).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    # 得到所有空位的action概率分布p和v，可以用于外部调用
    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    # 进行一次参数更新，根据调用传回来的winner_batch计算loss
    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(np.array(state_batch)).cuda())
            mcts_probs = Variable(torch.FloatTensor(np.array(mcts_probs)).cuda())
            winner_batch = Variable(torch.FloatTensor(np.array(winner_batch)).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(np.array(state_batch)))
            mcts_probs = Variable(torch.FloatTensor(np.array(mcts_probs)))
            winner_batch = Variable(torch.FloatTensor(np.array(winner_batch)))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        # return loss.data[0], entropy.data[0]
        #for pytorch version >= 0.5 please use the following line instead.
        return loss.item(), entropy.item()

    # 得到模型参数
    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    # 保存模型参数
    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
