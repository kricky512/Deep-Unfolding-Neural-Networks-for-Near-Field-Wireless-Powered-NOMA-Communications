from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class SystemConfig:
    N: int = 32
    K: int = 4
    fc_hz: float = 28e9
    alpha0: float = 0.3
    eta: float = 0.5
    delta: float = 0.8
    T: float = 1.0
    P: float = 1.0                  # W
    sigma2: float = 1e-13           # W，物理噪声方差
    nlos_paths: int = 3
    nlos_los_power_db: float = -10.0 # 路径系数总功率比；明确的仿真假设
    user_distance_m: tuple = (1.0, 3.0)
    user_angle_deg: tuple = (-30.0, 30.0)  # 相对阵列正前方 x 轴
    scatterer_distance_m: tuple = (0.5, 3.0)
    scatterer_angle_deg: tuple = (-70.0, 70.0)

    @property
    def wavelength(self):
        return 300000000.0 / self.fc_hz

    @property
    def antenna_spacing(self):
        return self.wavelength / 2

    @property
    def kappa(self):
        return self.alpha0 * self.eta * self.delta / (1 - self.alpha0)


@dataclass
class AOConfig:
    max_outer_iter: int = 30
    max_inner_iter: int = 20
    outer_tol: float = 1e-5
    inner_tol: float = 1e-5


@dataclass
class TrainingConfig:
    num_layers: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 50
    num_train: int = 1000
    num_val: int = 200
    num_test: int = 200
    seed: int = 2026
    device: str = "cpu"             # 或通过命令行 --device cuda
    dtype: str = "complex128"
    threads: int = 4
    grad_clip: float = 1.0


@dataclass
class Config:
    system: SystemConfig = field(default_factory=SystemConfig)
    ao: AOConfig = field(default_factory=AOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)

    @property
    def data_dir(self):
        return self.root / "data"

    @property
    def checkpoint_dir(self):
        return self.root / "checkpoints"

    @property
    def results_dir(self):
        return self.root / "results"

    def ensure_dirs(self):
        for path in (self.data_dir, self.checkpoint_dir, self.results_dir):
            path.mkdir(parents=True, exist_ok=True)

    def validate(self):
        s, a, t = self.system, self.ao, self.training
        if min(s.N, s.K) < 1 or s.nlos_paths < 0 or s.fc_hz <= 0:
            raise ValueError("N/K/fc 必须为正，NLoS 路径数不能为负。")
        if not (0 < s.alpha0 < 1 and 0 < s.eta <= 1 and 0 < s.delta <= 1):
            raise ValueError("alpha0/eta/delta 超出取值范围。")
        if min(s.P, s.sigma2, s.T) <= 0:
            raise ValueError("P、sigma2、T 必须为正；功率使用 W。")
        for interval in (s.user_distance_m, s.scatterer_distance_m):
            if interval[0] <= 0 or interval[1] < interval[0]:
                raise ValueError("距离区间必须为正且有序。")
        for interval in (s.user_angle_deg, s.scatterer_angle_deg):
            if not -90 < interval[0] <= interval[1] < 90:
                raise ValueError("本示例限定传播节点位于阵列前方，角度应在 (-90,90) 内。")
        if min(a.max_outer_iter, a.max_inner_iter) < 1 or min(a.outer_tol, a.inner_tol) <= 0:
            raise ValueError("迭代次数和收敛容差必须为正。")
        if t.num_layers < 2 or min(t.batch_size, t.num_epochs, t.num_train, t.num_val, t.num_test, t.threads) < 1:
            raise ValueError("至少两层网络，其余计数参数必须为正。")
        if t.learning_rate <= 0 or t.grad_clip <= 0:
            raise ValueError("学习率与梯度裁剪阈值必须为正。")
        if t.dtype not in ("complex64", "complex128"):
            raise ValueError("dtype 应为 complex64 或 complex128。")


def get_config():
    config = Config()
    config.validate()
    return config


def config_to_dict(config):
    result = asdict(config)
    result["root"] = str(config.root)
    return result


def add_common_arguments(parser):
    parser.add_argument("--smoke", action="store_true", help="快速小规模例子，输出到 results/smoke/")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--check", action="store_true", help="只检查配置，不进行训练或正式对比")


def config_from_arguments(args):
    config = get_config()
    if args.smoke:
        s, a, t = config.system, config.ao, config.training
        s.N, s.K, s.sigma2 = 8, 3, 1e-9
        s.user_distance_m = s.scatterer_distance_m = (0.12, 0.22)
        a.max_outer_iter, a.max_inner_iter = 6, 6
        t.num_layers, t.num_epochs, t.batch_size = 3, 3, 8
        t.num_train, t.num_val, t.num_test = 24, 8, 8
        config.root = config.root / "results" / "smoke"
    if args.device:
        config.training.device = args.device
    config.validate()
    return config
