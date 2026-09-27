import math

def control_payload(v,w,dt):
 if not all(math.isfinite(x) for x in (v,w,dt)) or dt<=0:raise ValueError('finite velocity / positive dt required')
 return {'linear_velocity':[0.,0.,-float(v)],'angular_velocity':[0.,float(w),0.],'time_step':float(dt)}

def update_history(history,current,length):
 if length<1:raise ValueError('positive context required')
 return ([current]*length if not history else [*history,current][-length:])
