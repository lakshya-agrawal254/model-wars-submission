"""Diagnostic logging only: no additional inference or control/state changes."""
from pathlib import Path
def instrument(path):
 s=path.read_text();s=s.replace('self.expected=None;self.last_position=None;self.runtime=None;self.diagnostic_errors=0','self.expected=None;self.last_position=None;self.runtime=None;self.diagnostic_errors=0\n        self.last_mode=None;self.last_stationary=None;self.raw_head=None')
 s=s.replace('self.runtime=Runtime()','self.runtime=Runtime()\n        def capture_head(module,inputs,output):\n            self.raw_head=output.detach().cpu().tolist()\n        self.runtime.model.head.register_forward_hook(capture_head)')
 s=s.replace('if self.callbacks==1 or self.callbacks%120==0:\n                self.evidence(\'processed_packet\'', 'stationary=p.velocity.x**2+p.velocity.y**2+p.velocity.z**2<100**2\n            log_now=self.callbacks==1 or self.callbacks%12==0 or mode!=self.last_mode or stationary!=self.last_stationary\n            if os.environ.get("V13_LOG_EVERY_CALLBACK")=="1":log_now=True\n            if log_now:\n                self.evidence(\'processed_packet\'')
 s=s.replace('observation13=obs,mode=MODES[mode],controls=controls,','observation13=obs,mode=MODES[mode],controls=controls,\n                    raw_head=self.raw_head,hidden_norm=float(self.runtime.hidden.norm()),\n                    stationary=stationary,callback_delta_seconds=obs[12]/60,')
 s=s.replace('self.last_position=position\n            return output','self.last_position=position\n            self.last_mode=mode;self.last_stationary=stationary\n            return output');path.write_text(s)
if __name__=='__main__':
 import sys
 instrument(Path(sys.argv[1]))
