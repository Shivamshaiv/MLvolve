import numpy as np, glob
from PIL import Image, ImageFilter
frames=sorted(glob.glob('allf/*.png'))
def load(i): return np.asarray(Image.open(frames[i]).convert('RGB'),np.float32)
cache={}
def F(i):
    if i not in cache: cache[i]=load(i)
    return cache[i]
def avg_crop(sec,box,thr=6.0):
    ref=int(sec*25); x0,y0,x1,y1=box; r=F(ref)[y0:y1,x0:x1]; acc=[]
    for j in range(max(0,ref-60),min(len(frames),ref+60)):
        c=F(j)[y0:y1,x0:x1]
        if np.sqrt(np.mean((c-r)**2))<thr: acc.append(c)
    print(box,len(acc)); return np.mean(acc,0)
def up(a,s): 
    im=Image.fromarray(np.clip(a,0,255).astype(np.uint8))
    im=im.resize((im.width*s,im.height*s),Image.LANCZOS)
    return im.filter(ImageFilter.UnsharpMask(radius=2,percent=60,threshold=2))
def key(im,lo=10,hi=45):
    a=np.asarray(im,np.float32); h,w,_=a.shape
    corners=np.concatenate([a[:4,:4].reshape(-1,3),a[:4,-4:].reshape(-1,3),a[-4:,:4].reshape(-1,3),a[-4:,-4:].reshape(-1,3)])
    bg=np.median(corners,0); d=np.sqrt(((a-bg)**2).sum(-1))
    al=np.clip((d-lo)/(hi-lo),0,1); al=al*al*(3-2*al)
    c=(a-(1-al[...,None])*bg)/np.maximum(al[...,None],1e-3); c=np.clip(c,0,255)
    out=np.dstack([c,al*255]).astype(np.uint8); return Image.fromarray(out,'RGBA')
jobs={'logo_prashant':(29.5,(90,135,628,252),3,True),'logo_itme':(5.5,(200,220,520,502),3,True),
'factory':(8.5,(85,222,634,534),2,False),'logo_pg':(14.5,(313,219,405,301),3,False),
'logo_pwp':(14.5,(253,543,467,622),3,True),'logo_ferber':(14.5,(253,883,467,929),3,False),
'map':(22.5,(30,200,692,582),2,True),'iso':(25.8,(248,1053,349,1130),3,False),'ce':(25.8,(376,1053,472,1130),3,True)}
names=['technical','suiting','denim','apparel','terry','sheeting','yarndyed','homefurn']
for i,n in enumerate(names):
    r,c=divmod(i,2); x=[114,450][c]; y=[214,454,694,934][r]
    jobs['app_'+n]=(19.5,(x,y,x+156,y+146),3,False)
for k,(sec,box,s,k_) in jobs.items():
    im=up(avg_crop(sec,box),s)
    (key(im) if k_ else im).save(f'assets/{k}.png')
