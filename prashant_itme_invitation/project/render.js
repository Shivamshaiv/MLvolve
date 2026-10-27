// Renders invitation.html frame-by-frame and pipes PNGs into ffmpeg (video-only intermediate).
// usage: node render.js m|w out.mp4 [fps]
const {chromium}=require('playwright'); const path=require('path'); const {spawn}=require('child_process');
(async()=>{
  const [f,out,fpsArg]=process.argv.slice(2); const fps=+(fpsArg||30);
  const vp=f==='w'?{width:1920,height:1080}:{width:1080,height:1920};
  const b=await chromium.launch(); const p=await b.newPage({viewport:vp});
  await p.goto('file://'+path.resolve(__dirname,'invitation.html')+'?f='+f); await p.evaluate(()=>window.ready);
  const dur=await p.evaluate(()=>window.DURATION); const n=Math.round(dur*fps);
  const ff=spawn('ffmpeg',['-y','-v','error','-f','image2pipe','-framerate',String(fps),'-c:v','png','-i','-',
    '-c:v','libx264','-preset','slow','-crf','12','-pix_fmt','yuv444p','-g',String(fps*2),out],{stdio:['pipe','inherit','inherit']});
  for(let i=0;i<n;i++){
    await p.evaluate(t=>render(t),i/fps);
    const buf=await p.screenshot({type:'png'});
    if(!ff.stdin.write(buf)) await new Promise(r=>ff.stdin.once('drain',r));
    if(i%150===0) console.log(f,i,'/',n);
  }
  ff.stdin.end(); await new Promise(r=>ff.on('close',r)); await b.close();
})();
