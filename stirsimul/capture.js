const maxImages = 960 * 2; // 60s * 16 fps * 10 min
const canvas = document.querySelector('canvas') //canvas is not defined in html but generated in light.js, so linkages are made here
const fixedWidth = 2048;
const fixedHeight = 1024;
var captureCount = 0;
var frameIndex = 0;
const zip = new JSZip();
const raw = zip.folder("raw");
const masked = zip.folder("masked");

window.captureCanvas = function(canvas){
    if (!canvas) {
      console.error("Main Canvas not found");
      return;
    }
    if (captureCount === 0) {
      if (frameIndex < maxImages) {
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
  
        leftCanvas.toBlob(rawBlob => {
          rightCanvas.toBlob(maskedBlob => {
            raw.file(`raw_${frameIndex + 1}.png`, rawBlob);
            masked.file(`masked_${frameIndex + 1}.png`, maskedBlob);
            frameIndex++;
            console.log("Capturing frames...");
            if (frameIndex % 100 === 0) {
              console.log(`Captured frame #${frameIndex}`);
            }
          }, "image/png");
        }, "image/png");
      }
      else {
        captureCount = 1;
        alert("Maximum number of captures reached, downloading now");
        bulkDownloadCaptures();
      }
    }
}    

  window.bulkDownloadCaptures = function(){
      console.log("Generating ZIP...");
    
      zip.generateAsync({ type: "blob" }).then(content => {
        const link = document.createElement("a");
        link.href = URL.createObjectURL(content);
        link.download = "captures.zip";
        link.click();
  
        // Reset data to free memory
        zip.remove("raw");
        zip.remove("masked");
        console.log("Captures cleared from memory.");
      });
  }