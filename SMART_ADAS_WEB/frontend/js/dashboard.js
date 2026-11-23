const $ = (id) => document.getElementById(id);
let selectedFile = null;
let previousStatus = '';
let previewLoading = false;

$('videoFile').addEventListener('change', event => {
  selectedFile = event.target.files[0] || null;
  $('fileName').textContent = selectedFile ? selectedFile.name : 'Chọn video từ máy tính';
  $('fileHint').textContent = selectedFile ? `${(selectedFile.size / 1024 / 1024).toFixed(1)} MB` : 'Chưa có video được tải lên';
});
async function api(path, method='GET', body=null) {
  const options={method}; if(body) options.body=body;
  const response=await fetch(path,options);
  const data=await response.json().catch(()=>({}));
  if(!response.ok) throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail||'Có lỗi xảy ra.'));
  return data;
}
function showMessage(message,error=false) {
  $('message').textContent=message;
  $('message').style.color=error?'#ff8490':'';
}
function setEditMode(enabled) { if(typeof window.setZoneEditMode==='function') window.setZoneEditMode(enabled); }
async function loadPreview(current) {
  if(previewLoading || typeof window.loadZonePreview!=='function') return;
  previewLoading=true;
  try { await window.loadZonePreview(current); }
  catch(error) { console.error('Không tải được ảnh chỉnh vùng:',error); }
  finally { previewLoading=false; }
}
$('uploadBtn').addEventListener('click',async()=>{
  if(!selectedFile) return showMessage('Hãy chọn video trước.',true);
  const form=new FormData();form.append('file',selectedFile);
  $('uploadBtn').disabled=true;showMessage('Đang tải video...');
  try {
    await api('/api/upload','POST',form);
    setEditMode(false); await loadPreview(false);
    $('startBtn').disabled=false;
    $('videoPlaceholder').style.display='flex';
    showMessage('Tải video thành công. Bạn có thể bắt đầu phân tích.');
  } catch(error) {showMessage(error.message,true);}
  finally {$('uploadBtn').disabled=false;}
});
$('startBtn').addEventListener('click',async()=>{
  try {
    await api('/api/start','POST');setEditMode(false);
    $('videoPlaceholder').style.display='none';
    $('startBtn').disabled=true;$('pauseBtn').disabled=false;$('resumeBtn').disabled=true;$('stopBtn').disabled=false;
    $('videoStream').src='/api/stream?ts='+Date.now();
  } catch(error) {showMessage(error.message,true);}
});
$('pauseBtn').addEventListener('click',async()=>{
  try {
    await api('/api/pause','POST');
    $('pauseBtn').disabled=true;$('resumeBtn').disabled=false;
    setEditMode(true); // Bật ngay, KHÔNG đợi preview
    void loadPreview(true);
  } catch(error) {showMessage(error.message,true);}
});
$('resumeBtn').addEventListener('click',async()=>{
  try {
    await api('/api/resume','POST');
    $('resumeBtn').disabled=true;$('pauseBtn').disabled=false;
    setEditMode(false);
  } catch(error) {showMessage(error.message,true);}
});
$('stopBtn').addEventListener('click',async()=>{
  try {await api('/api/stop','POST');$('stopBtn').disabled=true;setEditMode(false);}
  catch(error) {showMessage(error.message,true);}
});
function renderStatus(data) {
  const labels={idle:'SYSTEM IDLE',uploaded:'VIDEO READY',processing:'ANALYZING',paused:'PAUSED',completed:'COMPLETED',stopped:'STOPPED',error:'ERROR'};
  $('systemStatus').textContent=labels[data.status]||String(data.status).toUpperCase();
  $('message').textContent=data.error||data.message||'';
  $('frameInfo').textContent=`Frame: ${data.frame_index} / ${data.total_frames}`;
  $('fpsInfo').textContent=`Processing FPS: ${data.fps}`;
  $('progressInfo').textContent=`${data.progress}%`;
  $('progressBar').style.width=`${data.progress}%`;
  $('vehicleCount').textContent=data.vehicles;
  $('alertCount').textContent=data.alerts;
  $('fpsValue').textContent=data.fps;
  $('ttcValue').textContent=data.ttc==null?'—':`${data.ttc}s`;
  $('riskValue').textContent=data.risk;
  $('riskCard').className=`risk-card ${String(data.risk||'SAFE').toLowerCase()}`;
  const descriptions={SAFE:'Chưa phát hiện nguy cơ cao',WARNING:'Có đối tượng tiến gần vùng cảnh báo',DANGER:'Nguy cơ cao – cần chú ý'};
  $('riskDescription').textContent=descriptions[data.risk]||'';
  if(['completed','stopped','error'].includes(data.status)) {
    $('pauseBtn').disabled=true;$('resumeBtn').disabled=true;$('stopBtn').disabled=true;$('startBtn').disabled=false;
    setEditMode(false);
    if(data.status==='stopped') {
      $('videoStream').removeAttribute('src');$('videoPlaceholder').style.display='flex';
      if(previousStatus!=='stopped') void loadPreview(false);
    }
    if(data.status==='completed') {
      $('downloadBtn').classList.remove('disabled');
      $('downloadBtn').href='/api/download?ts='+Date.now();
    }
  }
  if(data.status==='processing') {
    if(previousStatus==='paused') setEditMode(false);
    $('videoPlaceholder').style.display='none';
    $('pauseBtn').disabled=false;$('resumeBtn').disabled=true;$('stopBtn').disabled=false;
  }
  if(data.status==='paused') {
    $('pauseBtn').disabled=true;$('resumeBtn').disabled=false;$('stopBtn').disabled=false;
    if(previousStatus!=='paused') setEditMode(true); // Chỉ bật khi vừa tạm dừng
    if(previousStatus!=='paused') void loadPreview(true);
  }
  if(data.status==='uploaded') $('startBtn').disabled=false;
  if(data.status==='error') showMessage(data.error||data.message,true);
  previousStatus=data.status;
}
async function refresh() {
  try {
    const data=await api('/api/status');renderStatus(data);
    const events=await api('/api/events');
    const tbody=$('eventTable');
    if(!events.length) tbody.innerHTML='<tr><td colspan="6" class="empty">Chưa có sự kiện cảnh báo.</td></tr>';
    else tbody.innerHTML=events.map(event=>`<tr><td>${event.time}</td><td>${event.video_time}s</td><td>#${event.track_id}</td><td>${event.object}</td><td><span class="pill ${event.risk}">${event.risk}</span></td><td>${event.description}</td></tr>`).join('');
  } catch(error) {showMessage(error.message,true);}
}
setInterval(refresh,1000);
refresh();
