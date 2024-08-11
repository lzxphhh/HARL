import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.self_attention_multi_head import Encoder  # multi head self attention
from harl.models.base.hierarchical_state_rep import Hierarchical_state_rep
from harl.models.base.prediction_rep import Prediction_rep
from harl.models.base.cross_aware_rep import Cross_aware_rep
from harl.models.base.rnn import RNNLayer
from harl.models.base.act import ACTLayer
from harl.utils.envs_tools import get_shape_from_obs_space
import yaml
import copy

class StochasticPolicy(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticPolicy model.
        Args:
            args: (dict) arguments containing relevant model information. #yaml里面的model和algo的config打包
            obs_space: (gym.Space) observation space.  # 单个智能体的观测空间 eg: Box (18,)
            action_space: (gym.Space) action space. # 单个智能体的动作空间 eg: Discrete(5,)
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(StochasticPolicy, self).__init__()
        # Load the environment arguments
        env_args = yaml.load(
            open('/home/spyder/projects/zhengxuan_projects/Mixed_traffic/HARL/harl/configs/envs_cfgs/bottleneck.yaml',
                 'r'),
            Loader=yaml.FullLoader)
        self.env_args = copy.deepcopy(env_args)
        self.strategy = env_args['strategy']
        self.hidden_sizes = args["hidden_sizes"]  # MLP隐藏层神经元数量
        self.args = args  # yaml里面的model和algo的config打包
        self.gain = args["gain"]  # 激活函数的斜率或增益，增益较大的激活函数会更敏感地响应输入的小变化，而增益较小的激活函数则会对输入的小变化不那么敏感
        self.initialization_method = args["initialization_method"]  # 网络权重初始化方法
        self.use_policy_active_masks = args["use_policy_active_masks"]  # TODO：这是什么

        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]  # TODO：二者的区别是什么
        self.use_recurrent_policy = args["use_recurrent_policy"]
        # number of recurrent layers
        self.recurrent_n = args["recurrent_n"]  # RNN层数
        self.tpdv = dict(dtype=torch.float32, device=device)  # dtype和device

        obs_shape = get_shape_from_obs_space(obs_space)  # 获取观测空间的形状，tuple of integer. eg: （18，）

        # 根据观测空间的形状，选择CNN或者MLP作为基础网络，用于base提取特征，输入大小obs_shape，输出大小hidden_sizes[-1]
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)
        self.attention = Encoder(obs_shape[0], action_space.n, 1, self.hidden_sizes[-1], 4, 'Discrete')
        self.hierarchical = Hierarchical_state_rep(obs_shape[0], action_space.n, self.hidden_sizes[-1], 'Discrete', args)
        self.prediction = Prediction_rep(obs_shape[0], action_space.n, self.hidden_sizes[-1], 'Discrete', args)
        self.cross_aware = Cross_aware_rep(obs_shape[0], action_space.n, self.hidden_sizes[-1], 'Discrete', args)
        self.num_CAVs = self.env_args['num_CAVs']
        self.num_HDVs = self.env_args['num_HDVs']
        self.CAV_ids = [f'CAV_{i}' for i in range(self.num_CAVs)]
        self.HDV_ids = [f'HDV_{i}' for i in range(self.num_HDVs)]
        self.veh_ids = self.CAV_ids + self.HDV_ids
        if self.strategy == 'prediction':
            self.prediction_output = {}
            self.prediction_error = {}
            for i in range(3):
                self.prediction_output[f'hist_{i + 1}'] = {veh_id: {pre_id:[] for pre_id in self.veh_ids} for veh_id in self.CAV_ids}
                self.prediction_error[f'hist_{i + 1}'] = {veh_id: {pre_id:[] for pre_id in self.veh_ids} for veh_id in self.CAV_ids}
        # 如果使用RNN，初始化RNN层 #TODO: 暂时还没看
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )

        # 初始化ACT层, 用于输出动作(动作概率)，输入大小hidden_sizes[-1]，输出大小action_space.n
        self.act = ACTLayer(
            action_space,
            self.hidden_sizes[-1],
            self.initialization_method,
            self.gain,
            args,  # yaml里面的model和algo的config打包
        )

        self.to(device)

    def forward(
            self, obs, rnn_states, masks, available_actions=None, deterministic=False
    ):
        """Compute actions from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
            action_log_probs: (torch.Tensor) log probabilities of taken actions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        # 检查输入的dtype和device是否正确，变形到在cuda上的tensor以方便进入网络
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        # 用base提取特征-输入大小obs_shape，输出大小hidden_sizes[-1], eg: TensorShape([20, 120]) 并行环境数量 x hidden_sizes[-1]
        env_num = obs.size(0)
        if self.strategy == 'base':
            actor_features = self.base(obs)
            prediction_error_output = torch.zeros(env_num, 1, device=self.tpdv['device'])
        elif self.strategy == 'attention':
            actor_features = self.attention(obs)
            prediction_error_output = torch.zeros(env_num, 1, device=self.tpdv['device'])
        elif self.strategy == 'hierarchical':
            actor_features = self.hierarchical(obs, batch_size=obs.size(0))
            prediction_error_output = torch.zeros(env_num, 1, device=self.tpdv['device'])
        elif self.strategy == 'prediction':
            actor_features, future_states, ego_id, prediction_groundtruth = self.prediction(obs, batch_size=obs.size(0))
            prediction_error_output = torch.zeros(env_num, 1, device=self.tpdv['device'])
            if ego_id:
                prediction_mse_error = {key: torch.zeros(env_num, 1, device=self.tpdv['device']) for key in self.veh_ids}
                prediction_mae_error = {key: torch.zeros(env_num, 1, device=self.tpdv['device']) for key in self.veh_ids}
                if len(ego_id) > 10:
                    print('ego_id:', ego_id)
                for veh_id in self.veh_ids:
                    if self.prediction_output['hist_1'][ego_id][veh_id] != [] and prediction_groundtruth[veh_id] != []:
                        valid_indices = self.prediction_output['hist_1'][ego_id][veh_id][:, 0, 0] != 0
                        if valid_indices.any():
                            self.prediction_error['hist_1'][ego_id][veh_id] = self.prediction_output['hist_1'][ego_id][veh_id][valid_indices, 0, :] - prediction_groundtruth[veh_id][valid_indices, :]
                            prediction_mse_error[veh_id][valid_indices] = torch.mean(
                                torch.pow(self.prediction_error['hist_1'][ego_id][veh_id], 2), dim=1, keepdim=True)
                            prediction_mae_error[veh_id][valid_indices] = torch.mean(
                                torch.abs(self.prediction_error['hist_1'][ego_id][veh_id]), dim=1, keepdim=True)
                    else:
                        self.prediction_error['hist_1'][ego_id][veh_id] = []
                    if self.prediction_output['hist_2'][ego_id][veh_id] != [] and prediction_groundtruth[veh_id] != []:
                        valid_indices = self.prediction_output['hist_2'][ego_id][veh_id][:, 1, 0] != 0
                        if valid_indices.any():
                            self.prediction_error['hist_2'][ego_id][veh_id] = self.prediction_output['hist_2'][ego_id][veh_id][valid_indices, 1, :] - prediction_groundtruth[veh_id][valid_indices, :]
                            prediction_mse_error[veh_id][valid_indices] = torch.mean(
                                torch.pow(self.prediction_error['hist_2'][ego_id][veh_id], 2), dim=1, keepdim=True)
                            prediction_mae_error[veh_id][valid_indices] = torch.mean(
                                torch.abs(self.prediction_error['hist_2'][ego_id][veh_id]), dim=1, keepdim=True)
                    else:
                        self.prediction_error['hist_2'][ego_id][veh_id] = []
                    if self.prediction_output['hist_3'][ego_id][veh_id] != [] and prediction_groundtruth[veh_id] != []:
                        valid_indices = self.prediction_output['hist_3'][ego_id][veh_id][:, 2, 0] != 0
                        if valid_indices.any():
                            self.prediction_error['hist_3'][ego_id][veh_id] = self.prediction_output['hist_3'][ego_id][veh_id][valid_indices, 2, :] - prediction_groundtruth[veh_id][valid_indices, :]
                            prediction_mse_error[veh_id][valid_indices] = torch.mean(
                                torch.pow(self.prediction_error['hist_3'][ego_id][veh_id], 2), dim=1, keepdim=True)
                            prediction_mae_error[veh_id][valid_indices] = torch.mean(
                                torch.abs(self.prediction_error['hist_3'][ego_id][veh_id]), dim=1, keepdim=True)
                    else:
                        self.prediction_error['hist_3'][ego_id][veh_id] = []
                for i in range(env_num):
                    error_cumulative = 0
                    veh_count = 0
                    for veh_id in self.veh_ids:
                        if prediction_mse_error[veh_id][i, 0] != 0 and veh_id != ego_id:
                            error_cumulative += prediction_mae_error[veh_id][i, 0]   # prediction_mse_error
                            veh_count += 1
                    if veh_count != 0:
                        prediction_error_output[i, 0] = error_cumulative / veh_count
                    else:
                        prediction_error_output[i, 0] = 0
                self.prediction_output['hist_3'][ego_id] = self.prediction_output['hist_2'][ego_id]
                self.prediction_output['hist_2'][ego_id] = self.prediction_output['hist_1'][ego_id]
                self.prediction_output['hist_1'][ego_id] = future_states


        elif self.strategy == 'cross_aware':
            actor_features = self.cross_aware(obs, batch_size=obs.size(0))

        # 如果使用RNN，将特征和RNN状态输入RNN层，得到新的特征和RNN状态
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # 将特征和可用动作输入ACT层，得到动作，动作概率
        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )

        #  TODO: 得到actions后进入prediction_rep中 得到预测的输出 然后计算mse和mae 计算loss--world model的思路

        return actions, action_log_probs, rnn_states, prediction_error_output

    def evaluate_actions(
            self, obs, rnn_states, action, masks, available_actions=None, active_masks=None
    ):
        """Compute action log probability, distribution entropy, and action distribution.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            action: (np.ndarray / torch.Tensor) actions whose entropy and log probability to evaluate.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            active_masks: (np.ndarray / torch.Tensor) denotes whether an agent is active or dead.
        Returns:
            action_log_probs: (torch.Tensor) log probabilities of the input actions.
            dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
            action_distribution: (torch.distributions) action distribution.
        """
        # 检查输入的dtype和device是否正确，变形到在cuda上的tensor以方便进入网络
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        # actor_features = self.base(obs)
        # actor_features = self.attention(obs)
        # actor_features = self.hierarchical(obs, batch_size=obs.size(0))
        if self.strategy == 'base':
            actor_features = self.base(obs)
        elif self.strategy == 'attention':
            actor_features = self.attention(obs)
        elif self.strategy == 'hierarchical':
            actor_features = self.hierarchical(obs, batch_size=obs.size(0))
        elif self.strategy == 'prediction':
            actor_features, future_states, ego_id, prediction_groundtruth = self.prediction(obs, batch_size=obs.size(0))
        elif self.strategy == 'cross_aware':
            actor_features = self.cross_aware(obs, batch_size=obs.size(0))

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy, action_distribution = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self.use_policy_active_masks else None,
        )

        return action_log_probs, dist_entropy, action_distribution
