#!/usr/bin/env python3
"""Download the exact public datasets used in the study; no malware binaries are fetched."""
import argparse,concurrent.futures,hashlib,os,requests
from pathlib import Path
SOURCES={
 'bodmas':'https://www.kaggle.com/api/v1/datasets/download/dhoogla/bodmas',
 'ember2018':'https://ember.elastic.co/ember_dataset_2018_2.tar.bz2'}
def download(name,out,workers=16,chunk=16*1024*1024):
 r=requests.get(SOURCES[name],headers={'Range':'bytes=0-0'},timeout=60);r.raise_for_status();url=r.url
 n=int(r.headers['Content-Range'].split('/')[-1]);parts=Path(str(out)+'.parts');parts.mkdir(exist_ok=True)
 def get(i):
  path=parts/f'{i:05d}'
  a=i*chunk;b=min(n-1,(i+1)*chunk-1)
  if path.exists() and path.stat().st_size==b-a+1:return i
  for attempt in range(6):
   try:
    z=requests.get(url,headers={'Range':f'bytes={a}-{b}'},timeout=180)
    if z.status_code==206 and len(z.content)==b-a+1:
     path.write_bytes(z.content);print('chunk',i,flush=True);return i
   except Exception as e:print('retry',i,attempt,e,flush=True)
  raise RuntimeError(f'chunk {i} failed')
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:list(ex.map(get,range((n+chunk-1)//chunk)))
 h=hashlib.sha256()
 with open(out,'wb') as f:
  for i in range((n+chunk-1)//chunk):
   b=(parts/f'{i:05d}').read_bytes();f.write(b);h.update(b)
 print(name,n,h.hexdigest())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('dataset',choices=SOURCES);p.add_argument('output');a=p.parse_args();download(a.dataset,a.output)
