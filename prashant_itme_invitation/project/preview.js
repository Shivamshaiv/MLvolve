// node preview.js m|w out_dir t1 t2 ...  -> PNG stills at given times
const {chromium}=require('playwright'); const path=require('path');
(async()=>{
  const [f,out,...ts]=process.argv.slice(2);
  const b=await chromium.launch(); const vp=f==='w'?{width:1920,height:1080}:{width:1080,height:1920};
  const p=await b.newPage({viewport:vp});
  await p.goto('file://'+path.resolve(__dirname,'invitation.html')+'?f='+f); await p.evaluate(()=>window.ready);
  for(const t of ts){await p.evaluate(t=>render(t),+t); await p.screenshot({path:`${out}/${f}_${t}.png`});}
  await b.close();
})();
