const maxImages = 960*10; // 60s * 16 fps * 10 min
const canvas = document.querySelector('canvas') //canvas is not defined in html but generated in light.js, so linkages are made here
const fixedWidth = 2048;
const fixedHeight = 1024;
var captureCount = 0;
const captures = [];

window.captureCanvas = function(canvas){
    if (!canvas) {
      console.error("Main Canvas not found");
      return;
    }
    if (captureCount === 0) {
      console.log("Capturing frames...");
      //offscreen canvas define and draw
      const offscreenCanvas = document.createElement('canvas'); 
      offscreenCanvas.width = canvas.width;
      offscreenCanvas.height = canvas.height;  
      const offscreenCtx = offscreenCanvas.getContext('2d');
      offscreenCtx.drawImage(canvas, 0, 0);
    
      //left offscreencanvas
      const leftCanvas = document.createElement('canvas');
      leftCanvas.width = fixedWidth / 2;
      leftCanvas.height = fixedHeight;
      const leftCtx = leftCanvas.getContext('2d');
      leftCtx.drawImage(offscreenCanvas, 0, 0, offscreenCanvas.width/2, offscreenCanvas.height, 0, 0, leftCanvas.width, leftCanvas.height);
  
      //right offscreencanvas with greyscale filter
      const rightCanvas = document.createElement('canvas');
      rightCanvas.width = fixedWidth / 2;
      rightCanvas.height = fixedHeight;
      const rightCtx = rightCanvas.getContext('2d');
      rightCtx.drawImage(offscreenCanvas, offscreenCanvas.width/2, 0, offscreenCanvas.width/2, offscreenCanvas.height, 0, 0, rightCanvas.width, rightCanvas.height);
  
      const dataURL = canvas.toDataURL('image/png');
      const rawdataURL = leftCanvas.toDataURL('image/png'); //Base64-encoded data URL
      const maskeddataURL = rightCanvas.toDataURL('image/png'); 
      captures.push({real: dataURL, raw: rawdataURL, masked: maskeddataURL}); // Push it into the captures array
      
      if(captures.length % 100 === 0) { 
        console.log(`Captured frame #${captures.length}`);
      }
      if(captures.length >= maxImages){
        alert("Maximum number of captures reached, downloading now");
        bulkDownloadCaptures();
        captureCount += 1;
      }
    }
  }
  
window.bulkDownloadCaptures = function(){
    let zip = new JSZip();
    let rawFolder = zip.folder("raw");
    let maskedFolder = zip.folder("masked");  
    captures.forEach((capture, index) => {
    const {raw, masked} = capture; 
       
    const rawbase64Data = raw.split(',')[1]; // Extract base64 data for raw
    const maskedbase64Data = masked.split(',')[1]; // Extract base64 data for masked
  
    const rawbinaryData = atob(rawbase64Data); // Decode Base64 data
    const maskedbinaryData = atob(maskedbase64Data);
  
    const rawarrayBuffer = new Uint8Array(rawbinaryData.length); 
    const maskedarrayBuffer = new Uint8Array(maskedbinaryData.length);
  
    for (let i = 0; i < rawbinaryData.length; i++) {
      rawarrayBuffer[i] = rawbinaryData.charCodeAt(i);
    }
        
    for (let i = 0; i < maskedbinaryData.length; i++) {
      maskedarrayBuffer[i] = maskedbinaryData.charCodeAt(i);
    }
        
    const rawBlob = new Blob([rawarrayBuffer], { type: "image/png" });   // Create blobs for raw and masked images
    const maskedBlob = new Blob([maskedarrayBuffer], { type: "image/png" });
      
    rawFolder.file(`raw_${index + 1}.png`, rawBlob); // Add images to respective folders
    maskedFolder.file(`masked_${index + 1}.png`, maskedBlob);
  });
  
    // Generate and download the zip file
    zip.generateAsync({ type: "blob" }).then((content) => {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(content);
    link.download = "captures.zip"; // Name of the zip file
    link.click();
    captures.length = 0; // Clear captures array after download
    console.log("Captures array cleared.");
    });
}
