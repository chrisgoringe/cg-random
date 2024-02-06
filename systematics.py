from .common import BaseNode

class SystematicBase(BaseNode):
    CATEGORY = "randoms"
    LAST = None
    def IS_CHANGED(self, **kwargs):
        return float("NaN")
    
    def func(self, minimum, maximum, step, restart, **kwargs):
        if self.LAST is None or restart=='yes': 
            self.LAST = minimum if step>0 else maximum
        else:
            self.LAST += step
            if self.LAST > maximum: self.LAST = minimum
            if self.LAST < minimum: self.LAST = maximum
        dp = kwargs.get("decimal_places", None)
        return (round(self.LAST, dp) if dp is not None else self.LAST,)

class SystematicInt(SystematicBase):
    RETURN_TYPES = ("INT",)
    REQUIRED = {"minimum": ("INT", {"default": 0}), 
                "maximum": ("INT", {"default": 100}), 
                "step": ("INT", {"default":1}),
                "restart": (["no","yes"], ) }
    
class SystematicFloat(SystematicBase):
    RETURN_TYPES = ("FLOAT",)
    REQUIRED = {"minimum": ("FLOAT", {"default": 0}), 
                "maximum": ("FLOAT", {"default": 1}), 
                "step": ("FLOAT", {"default":0.001}),
                "restart": (["no","yes"], ) }
    OPTIONAL = { "decimal_places": ("INT", {"default": 3, "min":0, "max":10}), }