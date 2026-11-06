# one copy of the first loft's functions; its data offset and two literals are private variables.
# mode calls: one call site per loft. mode loop: one call site, a loop over the lofts' small table
import re,sys
n,out,mode=sys.argv[1:4]
s=open(n).read()
a=s.index('fn alice_loft_0_ap'); b=s.index('@compute')
parts=re.split(r'(?m)^(?=fn )',s[a:b])
bases,rad={},{}
keep=[];order=[]
for p in parts:
    m=re.match(r'fn alice_loft_(\d+)',p)
    if m:
        k=int(m.group(1))
        q=re.search(r'alice_node_data\[clamp\((\d+) \+',p)
        if q and k not in bases: bases[k]=int(q.group(1))
        d=re.match(r'fn alice_loft_\d+_dome(\d)',p)
        if d:
            lit=re.search(r'max\(d \+ ([\d.]+), 0\.0\) / ([\d.]+)',p)
            rad[(k,int(d.group(1)))]=lit.group(1)
        big=re.search(r'bigf / ([\d.]+)',p)
        if big and re.match(r'fn alice_loft_\d+\(',p):
            rad[(k,2)]=big.group(1)
            fd=re.search(r'dist <= ([\d.]+)\) \{ psi = 1.0; \} else \{ if \(dist >= ([\d.]+)\)',p)
            rad[(k,3)],rad[(k,4)]=fd.group(1),fd.group(2)
        if k!=0: continue
        if big and re.match(r'fn alice_loft_\d+\(',p): p=p.replace(rad[(k,2)],'alice_r2').replace(rad[(k,3)],'alice_r3').replace(rad[(k,4)],'alice_r4')
        if d: p=p.replace(lit.group(0),f'max(d + alice_r{d.group(1)}, 0.0) / alice_r{d.group(1)}')
        p=re.sub(r'clamp\(0 \+ (\(.*?\)), (\d+), (\d+)\)',lambda m:f'clamp(alice_b + {m.group(1)}, alice_b + {m.group(2)}, alice_b + {m.group(3)})',p)
        keep.append(p)
    elif p.startswith('fn sdf_eval'):
        order=[int(x) for x in re.findall(r'let d\d+ = alice_loft_(\d+)\(p\);',p)]
        keep.append(('EVAL',p))
    else: keep.append(p)
def set_(k): return f'alice_b = {bases[k]}; alice_r0 = {rad[(k,0)]}; alice_r1 = {rad[(k,1)]}; alice_r2 = {rad[(k,2)]}; alice_r3 = {rad[(k,3)]}; alice_r4 = {rad[(k,4)]};'
out_parts=[]
for p in keep:
    if isinstance(p,tuple):
        p=p[1]
        if mode=='calls':
            p=re.sub(r'let (d\d+) = alice_loft_(\d+)\(p\);',lambda m:f'{set_(int(m.group(2)))} let {m.group(1)} = alice_loft_0(p);',p)
        else:
            N=len(order)
            tb=','.join(str(bases[k]) for k in order); t0=','.join(rad[(k,0)] for k in order); t1=','.join(rad[(k,1)] for k in order); t2=','.join(rad[(k,2)] for k in order); t3=','.join(rad[(k,3)] for k in order); t4=','.join(rad[(k,4)] for k in order)
            p=f'''fn sdf_eval(p: vec3<f32>) -> f32 {{
    var tb = array<i32,{N}>({tb});
    var t0 = array<f32,{N}>({t0});
    var t1 = array<f32,{N}>({t1});
    var t2 = array<f32,{N}>({t2});
    var t3 = array<f32,{N}>({t3});
    var t4 = array<f32,{N}>({t4});
    var d_ub: f32 = 0.0;
    for (var k = 0; k < {N}; k += 1) {{
        alice_b = tb[k]; alice_r0 = t0[k]; alice_r1 = t1[k]; alice_r2 = t2[k]; alice_r3 = t3[k]; alice_r4 = t4[k];
        let d = alice_loft_0(p);
        d_ub = select(d, min(d_ub, d), k > 0);
    }}
    return d_ub;
}}
'''
    out_parts.append(p)
r='var<private> alice_b: i32 = 0;\nvar<private> alice_r0: f32 = 0.0;\nvar<private> alice_r1: f32 = 0.0;\nvar<private> alice_r2: f32 = 0.0;\nvar<private> alice_r3: f32 = 0.0;\nvar<private> alice_r4: f32 = 0.0;\n'+''.join(out_parts)
assert 'clamp(0 +' not in r and 'alice_loft_2' not in r
open(out,'w').write(r)
print('CASE',out.split('/')[-1],len(r),'bytes',len(order),'lofts')
