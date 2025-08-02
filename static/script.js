let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let image = new Image();
let startX, startY, endX, endY;
let drawing = false;
let uploadedImageData = '';

document.getElementById('imageInput').addEventListener('change', function(e) {
  let reader = new FileReader();
  reader.onload = function(event) {
    image.onload = function() {
      canvas.width = image.width;
      canvas.height = image.height;
      ctx.drawImage(image, 0, 0);
    }
    image.src = event.target.result;
    uploadedImageData = event.target.result;
  }
  reader.readAsDataURL(e.target.files[0]);
});

canvas.addEventListener('mousedown', function(e) {
  const rect = canvas.getBoundingClientRect();
  startX = e.clientX - rect.left;
  startY = e.clientY - rect.top;
  drawing = true;
});

canvas.addEventListener('mouseup', function(e) {
  const rect = canvas.getBoundingClientRect();
  endX = e.clientX - rect.left;
  endY = e.clientY - rect.top;
  drawing = false;

  // Draw rectangle
  ctx.drawImage(image, 0, 0); // Clear previous boxes
  ctx.beginPath();
  ctx.rect(startX, startY, endX - startX, endY - startY);
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'red';
  ctx.stroke();
});

async function sendPrediction() {
  const box = [startX, startY, endX, endY];
  const res = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: uploadedImageData, box })
  });

  const data = await res.json();
  document.getElementById('result').src = data.mask;
}
