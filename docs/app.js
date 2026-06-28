const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const statusBadge = document.getElementById('status');
const apiKeyInput = document.getElementById('api-key');

let session;
let isDetecting = false;
let lastPostTime = 0;
const POST_INTERVAL_MS = 3000; // Giới hạn gửi dữ liệu 3 giây 1 lần để tránh sập server
const CONFIDENCE_THRESHOLD = 0.5; // Ngưỡng tự tin (50%)
const API_URL = "https://vehicle-api-v2p0.onrender.com/api/v1/vehicles"; 

// Khởi tạo mô hình ONNX
async function initModel() {
    try {
        statusBadge.textContent = "Loading AI Model...";
        statusBadge.className = "status-badge loading";
        
        // Cấu hình webassembly để chạy mượt trên chip điện thoại
        ort.env.wasm.numThreads = 4;
        session = await ort.InferenceSession.create('best.onnx', { executionProviders: ['wasm'] });
        
        statusBadge.textContent = "AI Ready";
        statusBadge.className = "status-badge ready";
        startBtn.disabled = false;
    } catch (e) {
        console.error(e);
        statusBadge.textContent = "Model Error";
        statusBadge.className = "status-badge error";
        alert("Lỗi tải mô hình AI! Hãy chắc chắn file best.onnx nằm cùng thư mục.");
    }
}

// Mở luồng Camera
startBtn.addEventListener('click', async () => {
    if (isDetecting) return;
    
    try {
        // Yêu cầu sử dụng Camera sau của điện thoại (facingMode: 'environment')
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 640 } },
            audio: false
        });
        video.srcObject = stream;
        video.play();
        
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            isDetecting = true;
            startBtn.disabled = true;
            startBtn.textContent = "Đang Quét...";
            requestAnimationFrame(detectFrame);
        };
    } catch (e) {
        console.error("Camera access denied", e);
        alert("Vui lòng cấp quyền truy cập Camera cho trình duyệt!");
    }
});

// Vòng lặp nhận diện liên tục
async function detectFrame() {
    if (!isDetecting) return;
    
    // Dùng 1 canvas nháp để lấy dữ liệu ảnh kích thước 640x640 (chuẩn YOLOv8)
    const offscreen = document.createElement('canvas');
    offscreen.width = 640;
    offscreen.height = 640;
    const offCtx = offscreen.getContext('2d');
    
    // Cắt và kéo giãn khung hình từ camera vào canvas 640x640
    offCtx.drawImage(video, 0, 0, 640, 640);
    const imgData = offCtx.getImageData(0, 0, 640, 640);
    
    // Tiền xử lý: Chuyển đổi pixel sang Float32Array [1, 3, 640, 640] chuẩn RGB
    const float32Data = new Float32Array(3 * 640 * 640);
    for (let i = 0; i < 640 * 640; i++) {
        // Chuẩn hóa màu từ 0-255 xuống 0.0-1.0
        float32Data[i] = imgData.data[i * 4] / 255.0;                   // Kênh Đỏ (R)
        float32Data[640 * 640 + i] = imgData.data[i * 4 + 1] / 255.0;       // Kênh Xanh Lục (G)
        float32Data[2 * 640 * 640 + i] = imgData.data[i * 4 + 2] / 255.0;   // Kênh Xanh Lam (B)
    }
    
    const tensor = new ort.Tensor('float32', float32Data, [1, 3, 640, 640]);
    
    try {
        // Cho AI ăn hình ảnh và chờ kết quả
        const results = await session.run({ 'images': tensor });
        const output = results['output0'].data; // Kết quả trả về kích thước [1, 9, 8400]
        
        processAndDraw(output, video.videoWidth, video.videoHeight);
    } catch (e) {
        console.error("Lỗi suy luận AI:", e);
    }
    
    // Lặp lại quy trình ở khung hình tiếp theo (siêu mượt 30fps)
    requestAnimationFrame(detectFrame);
}

// Xử lý kết quả và vẽ lên màn hình
function processAndDraw(output, imgW, imgH) {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Xóa khung cũ
    let detected = false;
    let maxConf = 0;
    let bestClass = 1;
    
    // YOLOv8 sinh ra 8400 hộp dự đoán. Ta duyệt để tìm hộp có độ chính xác cao nhất.
    for (let i = 0; i < 8400; i++) {
        let maxClassConf = 0;
        let classId = -1;
        // Có 5 class (từ index 4 đến 8)
        for (let c = 0; c < 5; c++) {
            let conf = output[(4 + c) * 8400 + i];
            if (conf > maxClassConf) {
                maxClassConf = conf;
                classId = c;
            }
        }
        
        // Nếu AI tin tưởng đây là xe (trên 50%)
        if (maxClassConf > CONFIDENCE_THRESHOLD) {
            detected = true;
            if (maxClassConf > maxConf) {
                maxConf = maxClassConf;
                bestClass = classId; // Loại xe
            }
            
            // Lấy tọa độ (Tâm X, Tâm Y, Rộng, Dài)
            const xc = output[0 * 8400 + i];
            const yc = output[1 * 8400 + i];
            const w = output[2 * 8400 + i];
            const h = output[3 * 8400 + i];
            
            // Quy đổi ngược về kích thước thật của Camera
            const x1 = (xc - w / 2) * (imgW / 640);
            const y1 = (yc - h / 2) * (imgH / 640);
            const boxW = w * (imgW / 640);
            const boxH = h * (imgH / 640);
            
            // Vẽ hộp xanh lá cây bao quanh xe
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 4;
            ctx.strokeRect(x1, y1, boxW, boxH);
            
            // Hiển thị text độ chính xác
            ctx.fillStyle = '#00ff00';
            ctx.font = 'bold 18px Arial';
            ctx.fillText(`Xe cộ: ${Math.round(maxClassConf * 100)}%`, x1, y1 > 20 ? y1 - 10 : y1 + 20);
        }
    }
    
    // Nếu có xe, bắn Data lên Server
    if (detected) {
        sendDataToCloud(bestClass, maxConf);
    }
}

// Bắn Data lên Server Render (Zero-Cost)
async function sendDataToCloud(category_id, confidence) {
    const now = Date.now();
    // Chặn gửi rác (Ví dụ xe dừng đèn đỏ 1 phút thì không gửi 60 lần)
    if (now - lastPostTime < POST_INTERVAL_MS) return; 
    
    const apiKey = apiKeyInput.value.trim();
    if (!apiKey) return; // Chưa nhập mật khẩu thì không gửi
    
    lastPostTime = now;
    
    try {
        // Gửi ngầm qua mạng bằng Fetch API
        await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': apiKey
            },
            body: JSON.stringify({
                detected_at: new Date().toISOString(),
                category_id: category_id + 1, // Fix DB ID (0-indexed -> 1-indexed)
                confidence: confidence
            })
        });
        console.log("Đã gửi báo cáo lên Máy chủ!");
    } catch (e) {
        console.error("Lỗi khi gửi lên Máy chủ:", e);
    }
}

// Chạy luôn từ lúc mở web
initModel();
