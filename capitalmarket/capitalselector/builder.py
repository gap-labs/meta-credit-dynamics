from .core import CapitalSelector, Channel
from .channels import DummyChannel
from .stats import EWMAStats
from .rebirth import RebirthPolicy
from .reweight import exp_reweight
from .config import ProfileAConfig, ProfileBConfig
from .phase_i_state import DEFAULT_LAMBDA_RISK, validate_lambda_risk
from .selector_policy import DEFAULT_SELECTOR_POLICY, SelectorPolicy, validate_selector_policy


class CapitalSelectorBuilder:
    def __init__(self):
        self._wealth = 1.0
        self._rebirth_threshold = 0.5
        self._beta = 0.01
        self._eta = 1.0
        self._kind = "entrepreneur"
        self._rebirth_policy = None
        self._channels: list[Channel] = []
        self._selector_policy: SelectorPolicy = DEFAULT_SELECTOR_POLICY
        self._lambda_risk: float = DEFAULT_LAMBDA_RISK
        self._resolved_config: dict[str, object] = {}

    @classmethod
    def from_profile(cls, profile: ProfileAConfig | ProfileBConfig):
        profile.validate_closed()
        builder = cls()
        builder._resolved_config = {
            "dt": profile.dt,
            "cost_distribution": profile.cost_distribution,
            "score_mode": profile.score_mode,
            "stats_signal": profile.stats_signal,
            "stack_weighting": profile.stack_weighting,
            "freeze_stats": profile.freeze_stats,
            "credit_condition_active": profile.credit_condition_active,
            "sparsity_active": profile.sparsity_active,
            "rebirth_pool_active": profile.rebirth_pool_active,
        }
        return builder

    def with_initial_wealth(self, w: float):
        self._wealth = float(w); return self

    def with_rebirth_threshold(self, t: float):
        self._rebirth_threshold = float(t); return self

    def with_stats(self, beta: float):
        self._beta = float(beta); return self

    def with_reweight_eta(self, eta: float):
        self._eta = float(eta); return self

    def with_kind(self, kind: str):
        self._kind = kind; return self

    def with_rebirth_policy(self, policy: RebirthPolicy):
        self._rebirth_policy = policy; return self

    def with_selector_policy(self, selector_policy: SelectorPolicy | str):
        self._selector_policy = validate_selector_policy(str(selector_policy)); return self

    def with_lambda_risk(self, lambda_risk: float):
        self._lambda_risk = validate_lambda_risk(lambda_risk); return self

    def with_channels(self, channels: list[Channel]):
        self._channels = channels; return self

    def with_K(self, K: int):
        """Setzt die Simplex-Dimension über Dummy-Kanäle ohne Semantik."""
        K = int(K)
        if K < 0:
            raise ValueError("K must be >= 0")
        self._channels = [DummyChannel() for _ in range(K)]
        return self

    def build(self) -> CapitalSelector:
        stats = EWMAStats(beta=self._beta, seed_var=1.0)

        def reweight(w, adv):
            return exp_reweight(w, adv, self._eta)

        return CapitalSelector(
            wealth=self._wealth,
            rebirth_threshold=self._rebirth_threshold,
            stats=stats,
            reweight_fn=reweight,
            kind=self._kind,
            rebirth_policy=self._rebirth_policy,
            channels=self._channels,
            selector_policy=self._selector_policy,
            lambda_risk=self._lambda_risk,
        )
