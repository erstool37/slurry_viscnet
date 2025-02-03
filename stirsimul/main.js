/*
 * WebGL Water
 * http://madebyevan.com/webgl-water/
 *
 * Copyright 2011 Evan Wallace
 * Released under the MIT license
 */

function text2html(text) {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>');
}

function handleError(text) {
  var html = text2html(text);
  if (html == 'WebGL not supported') {
    html = 'Your browser does not support WebGL.<br>Please see\
    <a href="http://www.khronos.org/webgl/wiki/Getting_a_WebGL_Implementation">\
    Getting a WebGL Implementation</a>.';
  }
  var loading = document.getElementById('loading');
  loading.innerHTML = html;
  loading.style.zIndex = 1;
}
window.onerror = handleError;

var gl = GL.create();
var water;
var cubemap;
var renderer;
var angleX = -90; //camera angleX
var angleY = 0;  //camera angleY
var useSpherePhysics = false;
var center;
var oldCenter;
var velocity;
var gravity;
var radius;
var paused = false;

//After window loading, the following function will be executed
window.onload = function () {
  var ratio = window.devicePixelRatio || 1;
  var help = document.getElementById('help');

  //canvas and viewport definition
  function onresize() {
    var width = innerWidth;
    var height = innerHeight;
    gl.canvas.width = width * ratio;
    gl.canvas.height = height * ratio;
    gl.canvas.style.width = width + 'px';
    gl.canvas.style.height = height + 'px';

    //left viewport define
    gl.viewport(0, 0, gl.canvas.width / 2, gl.canvas.height);
    gl.matrixMode(gl.PROJECTION);
    gl.loadIdentity();
    gl.perspective(30, (gl.canvas.width / 2) / gl.canvas.height, 0.01, 100);
    gl.matrixMode(gl.MODELVIEW);

    //right viewport define
    gl.viewport(gl.canvas.width / 2, 0, gl.canvas.width / 2, gl.canvas.height);
    gl.matrixMode(gl.PROJECTION);
    gl.loadIdentity();
    gl.perspective(30, (gl.canvas.width / 2) / gl.canvas.height, 0.01, 100);
    gl.matrixMode(gl.MODELVIEW);
  } 

  //background color settings
  document.body.appendChild(gl.canvas);
  gl.clearColor(1, 1, 1, 1);
  water = new Water();
  renderer = new Renderer();
  cubemap = new Cubemap({
    xneg: document.getElementById('xneg'),
    xpos: document.getElementById('xpos'),
    yneg: document.getElementById('ypos'),
    ypos: document.getElementById('ypos'),
    zneg: document.getElementById('zneg'),
    zpos: document.getElementById('zpos')
  });

  if (!water.textureA.canDrawTo() || !water.textureB.canDrawTo()) {
    throw new Error('Rendering to floating-point textures is required but not supported');
  }

  // starting sphere physics settings
  center = oldCenter = new GL.Vector(-0.0, 0.0, 4.0);  //y axis is considered as height, and z axis is the depth
  velocity = new GL.Vector();
  gravity = new GL.Vector(0, -2.0, 0);
  radius = 0.05;

  document.getElementById('loading').innerHTML = '';
  onresize();

  var requestAnimationFrame =
    window.requestAnimationFrame ||
    window.webkitRequestAnimationFrame ||
    function (callback) { setTimeout(callback, 0); };

  var prevTime = Date.now();
  var startTime = Date.now();
  var Duration = 60000; 
  var dropInterval = 1000;
  var lastDropTime = startTime;
  var lastCaptureTime = startTime;
  var captureInterval = 62.5; // 16 frames per second = 1000/16 = 62.5 ms per frame
  var captureCount = 0;

  function animate() {
      var nextTime = Date.now();
      var timeGap = nextTime - startTime;
      if (!paused) {
        update((nextTime - prevTime) / 1000);
        drawLeft();
        drawRight();
        if (timeGap < Duration && nextTime - lastDropTime >= dropInterval) {
            water.addDrop(Math.random() * 2 - 1, Math.random() * 2 - 1, 0.04, 0.1);
            lastDropTime = nextTime;
        }
        if (nextTime - lastCaptureTime >= captureInterval && captureCount === 0) {
          lastCaptureTime = nextTime;
          captureCanvas(document.querySelector('canvas')); // Store data URL in captures array
        }
      }
      prevTime = nextTime;
      requestAnimationFrame(animate);
    }
    
  requestAnimationFrame(animate); //animating through using 1)update, 2)draw
  window.onresize = onresize; //resizing window adjustments

  var prevHit;
  var planeNormal;
  var mode = -1;
  var MODE_ADD_DROPS = 0;
  var MODE_MOVE_SPHERE = 1;
  var MODE_ORBIT_CAMERA = 2;

  var oldX, oldY;

  function startDrag(x, y) {
    oldX = x;
    oldY = y;
    var tracer = new GL.Raytracer();
    var ray = tracer.getRayForPixel(x * ratio, y * ratio);
    var pointOnPlane = tracer.eye.add(ray.multiply(-tracer.eye.y / ray.y));
    var sphereHitTest = GL.Raytracer.hitTestSphere(tracer.eye, ray, center, radius);
    if (sphereHitTest) {
      mode = MODE_MOVE_SPHERE;
      prevHit = sphereHitTest.hit;
      planeNormal = tracer.getRayForPixel(gl.canvas.width / 2, gl.canvas.height / 2).negative();
    } else if (Math.abs(pointOnPlane.x) < 1 && Math.abs(pointOnPlane.z) < 1) {
      mode = MODE_ADD_DROPS;
      duringDrag(x, y);
    } else {
      mode = MODE_ORBIT_CAMERA;
    }
  }

  //mouse drag event defined
  function duringDrag(x, y) {
    switch (mode) {
      case MODE_ADD_DROPS: {
        var tracer = new GL.Raytracer();
        var ray = tracer.getRayForPixel(x * ratio, y * ratio);
        var pointOnPlane = tracer.eye.add(ray.multiply(-tracer.eye.y / ray.y));
        water.addDrop(pointOnPlane.x, pointOnPlane.z, 0.04, 0.1);//b, a is radius and strength
        if (paused) {
          water.updateNormals();
          renderer.updateCaustics(water);
        }
        break;
      }
      case MODE_MOVE_SPHERE: {
        var tracer = new GL.Raytracer();
        var ray = tracer.getRayForPixel(x * ratio, y * ratio);
        var t = -planeNormal.dot(tracer.eye.subtract(prevHit)) / planeNormal.dot(ray);
        var nextHit = tracer.eye.add(ray.multiply(t));
        center = center.add(nextHit.subtract(prevHit));
        center.x = Math.max(radius - 1, Math.min(1 - radius, center.x));
        center.y = Math.max(radius - 1, Math.min(10, center.y));
        center.z = Math.max(radius - 1, Math.min(1 - radius, center.z));
        prevHit = nextHit;
        if (paused) renderer.updateCaustics(water);
        break;
      }
      case MODE_ORBIT_CAMERA: {
        angleY -= x - oldX;
        angleX -= y - oldY;
        angleX = Math.max(-89.999, Math.min(89.999, angleX));
        break;
      }
    }
    oldX = x;
    oldY = y;
    if (paused) {
      drawLeft();
      drawRight();
    }
  }

  function stopDrag() {
    mode = -1;
  }

  function isHelpElement(element) {
    return element === help || element.parentNode && isHelpElement(element.parentNode);
  }

  document.onmousedown = function (e) {
    if (!isHelpElement(e.target)) {
      e.preventDefault();
      startDrag(e.pageX, e.pageY);
    }
  };

  document.onmousemove = function (e) {
    duringDrag(e.pageX, e.pageY);
  };

  document.onmouseup = function () {
    stopDrag();
  };

  document.ontouchstart = function (e) {
    if (e.touches.length === 1 && !isHelpElement(e.target)) {
      e.preventDefault();
      startDrag(e.touches[0].pageX, e.touches[0].pageY);
    }
  };

  document.ontouchmove = function (e) {
    if (e.touches.length === 1) {
      duringDrag(e.touches[0].pageX, e.touches[0].pageY);
    }
  };

  document.ontouchend = function (e) {
    if (e.touches.length == 0) {
      stopDrag();
    }
  };

  document.onkeydown = function (e) {
    if (e.which == ' '.charCodeAt(0)) paused = !paused;
    else if (e.which == 'G'.charCodeAt(0)) useSpherePhysics = !useSpherePhysics;
    else if (e.which == 'L'.charCodeAt(0) && paused) drawLeft(); drawRight();
  };

  var frame = 0;

  function update(seconds) {
    if (seconds > 1) return;
    frame += seconds * 2;
    if (mode == MODE_MOVE_SPHERE) {
      // Start from rest when the player releases the mouse after moving the sphere
      velocity = new GL.Vector();
    } else if (useSpherePhysics) {
      // Fall down with viscosity under water
      var percentUnderWater = Math.max(0, Math.min(1, (radius - center.y) / (2 * radius)));
      velocity = velocity.add(gravity.multiply(seconds - 1.1 * seconds * percentUnderWater));
      velocity = velocity.subtract(velocity.unit().multiply(percentUnderWater * seconds * velocity.dot(velocity)));
      center = center.add(velocity.multiply(seconds));
      // Bounce off the bottom
      if (center.y < radius - 1) {
        center.y = radius - 1;
        velocity.y = Math.abs(velocity.y) * 0.7;
      }
    }

    // Displace water around the sphere
    water.moveSphere(oldCenter, center, radius);
    oldCenter = center;

    // Update the water simulation and graphics
    water.stepSimulation();
    water.stepSimulation();
    water.updateNormals();
    renderer.updateCaustics(water);
  }

function drawLeft() {
  gl.viewport(0, 0, gl.canvas.width / 2, gl.canvas.height);

  // Change the light direction to the camera look vector where the L key is pressed
  if (GL.keys.L) {
    renderer.lightDir = GL.Vector.fromAngles((90 - angleY) * Math.PI / 180, -angleX * Math.PI / 180);
    if (paused) renderer.updateCaustics(water);
  }

  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.loadIdentity();
  gl.translate(0, 0, -4); //camera position, ensure fitting within viewport
  gl.rotate(-angleX, 1, 0, 0); //camera angleX
  gl.rotate(-angleY, 0, 1, 0); //camera angleY
  gl.translate(0, 0.5, 0); //camera position, fix after rotation

  gl.enable(gl.DEPTH_TEST); //camera depth buffer(distance from the camera) testing
  renderer.sphereCenter = center;
  renderer.sphereRadius = radius;
  renderer.renderCube(); //render cube
  renderer.renderWater(water, cubemap); //render water
  renderer.renderSphere(); //render sphere
  gl.disable(gl.DEPTH_TEST);
}

function drawRight() {
  gl.viewport(gl.canvas.width / 2, 0, gl.canvas.width / 2, gl.canvas.height);

  // Change the light direction to the camera look vector where the L key is pressed
  if (GL.keys.L) {
    renderer.lightDir = GL.Vector.fromAngles((90 - angleY) * Math.PI / 180, -angleX * Math.PI / 180);
    if (paused) renderer.updateCaustics(water);
  }

  gl.loadIdentity();
  gl.translate(0, 0, -4);
  gl.rotate(-angleX, 1, 0, 0);
  gl.rotate(-angleY, 0, 1, 0);
  gl.translate(0, 0.5, 0);

  gl.enable(gl.DEPTH_TEST);
  renderer.sphereCenter = center;
  renderer.sphereRadius = radius;
  renderer.renderCube();
  renderer.renderWaterMask(water, cubemap); //Masked rendered Water
  renderer.renderSphere();
  gl.disable(gl.DEPTH_TEST);
  }
}