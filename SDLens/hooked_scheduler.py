import inspect
from diffusers import DDPMScheduler

class HookedNoiseScheduler:
    scheduler: DDPMScheduler
    pre_hooks: list
    post_hooks: list

    def __init__(self, scheduler):
        object.__setattr__(self, 'scheduler', scheduler)
        object.__setattr__(self, 'pre_hooks', [])
        object.__setattr__(self, 'post_hooks', [])
    
    def step(self, model_output, timestep, sample, generator=None, return_dict=False, **kwargs):
        # diffusers 不同 scheduler 的 step(...) 参数不同（例如 PNDM 不接受 generator）。
        # 这里按底层 scheduler.step 的签名过滤 kwargs，避免传入不支持的参数导致 TypeError。
        if return_dict is not False:
            raise AssertionError("return_dict == True is not implemented")

        for hook in self.pre_hooks:
            hook_output = hook(model_output, timestep, sample, generator)
            if hook_output is not None:
                model_output, timestep, sample, generator = hook_output

        sig = inspect.signature(self.scheduler.step)
        params = sig.parameters
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        call_kwargs = dict(kwargs)
        if generator is not None:
            call_kwargs["generator"] = generator
        call_kwargs["return_dict"] = return_dict

        if not accepts_var_kw:
            call_kwargs = {k: v for k, v in call_kwargs.items() if k in params}

        out = self.scheduler.step(model_output, timestep, sample, **call_kwargs)

        # return_dict=False 时，diffusers 可能返回 tuple，也可能返回 SchedulerOutput-like 对象
        if isinstance(out, tuple):
            pred_prev_sample = out[0]
        elif hasattr(out, "prev_sample"):
            pred_prev_sample = out.prev_sample
        elif isinstance(out, dict) and "prev_sample" in out:
            pred_prev_sample = out["prev_sample"]
        else:
            raise TypeError(f"Unexpected scheduler.step output type: {type(out)}")

        for hook in self.post_hooks:
            hook_output = hook(pred_prev_sample)
            if hook_output is not None:
                pred_prev_sample = hook_output

        return (pred_prev_sample,)

    def __getattr__(self, name):
        return getattr(self.scheduler, name)

    def __setattr__(self, name, value):
        if name in {'scheduler', 'pre_hooks', 'post_hooks'}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.scheduler, name, value)
