(() => {
  const $ = id => document.getElementById(id);
  const canvas = $('zoneCanvas'), ctx = canvas.getContext('2d');
  const img = $('previewImage');
  const defaults = {
    warning: [[.475,.52],[.515,.52],[.610,.76],[.395,.76]],
    danger: [[.455,.65],[.555,.65],[.610,.76],[.395,.76]]
  };
  let zones = structuredClone(defaults), active = 'warning', dragging = -1, hasPreview = false;
  const message = (t, err=false) => { $('zoneMessage').textContent = t; $('zoneMessage').style.color = err ? '#ff8490' : ''; };
  function resize() {
    if (!img.naturalWidth) return;
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    draw();
  }
  function polygon(name, fill, stroke) {
    const pts = zones[name];
    ctx.beginPath();
    pts.forEach(([x,y],i) => i ? ctx.lineTo(x*canvas.width,y*canvas.height) : ctx.moveTo(x*canvas.width,y*canvas.height));
    ctx.closePath(); ctx.fillStyle = fill; ctx.strokeStyle = stroke; ctx.lineWidth = name === active ? 3 : 2;
    ctx.fill(); ctx.stroke();
    if (name === active) pts.forEach(([x,y],i) => {
      ctx.beginPath(); ctx.arc(x*canvas.width,y*canvas.height,8,0,2*Math.PI);
      ctx.fillStyle='#fff';ctx.fill();ctx.lineWidth=3;ctx.strokeStyle=stroke;ctx.stroke();
      ctx.font='bold 12px sans-serif';ctx.fillStyle='#fff';ctx.fillText(String(i+1),x*canvas.width+11,y*canvas.height-9);
    });
  }
  function draw() {
    ctx.clearRect(0,0,canvas.width,canvas.height);
    if (!hasPreview) return;
    polygon('warning','rgba(255,194,46,.13)','#ffca40');
    polygon('danger','rgba(255,60,70,.19)','#ff455b');
  }
  function pos(e) {
    const r=canvas.getBoundingClientRect();
    return [(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height];
  }
  function near(p) {
    let best=-1, dist=Infinity;
    zones[active].forEach((q,i) => {
      const d=Math.hypot((p[0]-q[0])*canvas.width,(p[1]-q[1])*canvas.height);
      if (d<dist) {dist=d;best=i;}
    });
    return dist<=23?best:-1;
  }
  canvas.addEventListener('pointerdown', e => {
    if (!hasPreview) return;
    dragging=near(pos(e));
    if (dragging>=0) {canvas.setPointerCapture(e.pointerId);e.preventDefault();}
  });
  canvas.addEventListener('pointermove', e => {
    if (dragging<0) return;
    const p=pos(e);
    zones[active][dragging]=p.map(v=>Math.max(0,Math.min(1,Math.round(v*10000)/10000)));
    draw();
    message('Đã thay đổi vùng. Nhấn Lưu vùng cảm biến trước khi phân tích.');
  });
  for (const event of ['pointerup','pointercancel','lostpointercapture']) canvas.addEventListener(event,()=>dragging=-1);
  function select(name) {
    active=name;
    $('editWarning').classList.toggle('editor-selected',name==='warning');
    $('editDanger').classList.toggle('editor-selected',name==='danger');
    draw();
  }
  $('editWarning').onclick=()=>select('warning');
  $('editDanger').onclick=()=>select('danger');
  $('resetZones').onclick=()=>{zones=structuredClone(defaults);draw();message('Đã đặt lại mặc định; nhấn Lưu để áp dụng.');};
  $('saveZones').onclick=async()=>{
    try {
      const r=await fetch('/api/zones',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(zones)});
      const data=await r.json();
      if (!r.ok) throw Error(typeof data.detail==='string'?data.detail:'Cấu hình không hợp lệ');
      message('Đã lưu! Vùng mới sẽ được sử dụng từ lần bắt đầu phân tích tiếp theo.');
    } catch(e) {message(e.message,true);}
  };
  window.loadZonePreview=async()=>{
    try {
      const r=await fetch('/api/preview?ts='+Date.now());
      if (!r.ok) throw Error('Không tải được ảnh xem trước');
      const blob=await r.blob();
      const url=URL.createObjectURL(blob);
      img.onload=()=>{hasPreview=true;$('previewHint').classList.add('hidden');resize();URL.revokeObjectURL(url);};
      img.src=url;
    } catch(e) {message(e.message,true);}
  };
  fetch('/api/zones').then(r=>r.json()).then(data=>{zones=data;draw();}).catch(()=>message('Không đọc được cấu hình vùng.',true));
  window.addEventListener('resize',resize);
})();
