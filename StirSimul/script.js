// Get the WebGL context
const canvas = document.getElementById("glCanvas");
const gl = canvas.getContext("webgl");

if (!gl) {
  console.error("WebGL not supported.");
}

// Resize the canvas to fit the window
resizeCanvas();
window.addEventListener("resize", resizeCanvas);

function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  gl.viewport(0, 0, canvas.width, canvas.height);
}

// Vertex shader: Transforms 3D vertices into 2D clip space
const vertexShaderSource = `
attribute vec3 a_Position;
attribute vec3 a_Normal;

uniform mat4 u_ModelMatrix;
uniform mat4 u_ViewMatrix;
uniform mat4 u_ProjectionMatrix;

varying vec3 v_Normal;
varying vec3 v_Position;

void main() {
    gl_Position = u_ProjectionMatrix * u_ViewMatrix * u_ModelMatrix * vec4(a_Position, 1.0);
    v_Normal = mat3(u_ModelMatrix) * a_Normal;
    v_Position = vec3(u_ModelMatrix * vec4(a_Position, 1.0));
}
`;

// Fragment shader: Simulates lighting for the cylinder and fluid
const fragmentShaderSource = `
precision mediump float;

varying vec3 v_Normal;
varying vec3 v_Position;

uniform vec3 u_LightPosition;
uniform vec3 u_ViewPosition;
uniform vec4 u_Color;
uniform float u_Transparency;

void main() {
    vec3 normal = normalize(v_Normal);
    vec3 lightDir = normalize(u_LightPosition - v_Position);
    vec3 viewDir = normalize(u_ViewPosition - v_Position);

    // Ambient lighting
    vec3 ambient = 0.2 * u_Color.rgb;

    // Diffuse lighting
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * u_Color.rgb;

    // Specular lighting
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16.0);
    vec3 specular = 0.3 * spec * vec3(1.0);

    // Combine lighting
    vec3 lighting = ambient + diffuse + specular;

    // Apply transparency
    gl_FragColor = vec4(lighting, u_Transparency);
}
`;

// Compile shaders and link program
function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function createProgram(gl, vertexSource, fragmentSource) {
  const vs = createShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fs = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);

  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }
  return program;
}

const program = createProgram(gl, vertexShaderSource, fragmentShaderSource);
gl.useProgram(program);

// Define cylinder parameters
const height = 0.5; // Height of the cylinder
const radius = 0.25; // Radius of the cylinder
const fluidHeight = 0.3; // Height of the fluid
const segments = 64; // Number of segments for the cylinder

// Generate vertices and normals
function generateCylinder(radius, height, segments) {
  const vertices = [];
  const normals = [];

  for (let i = 0; i <= segments; i++) {
    const theta = (i / segments) * Math.PI * 2;
    const x = Math.cos(theta) * radius;
    const z = Math.sin(theta) * radius;

    // Top and bottom
    vertices.push(x, height / 2, z, x, -height / 2, z);
    normals.push(x, 0, z, x, 0, z);
  }

  return { vertices, normals };
}

function generateDisk(radius, y, segments) {
  const vertices = [];
  const normals = [];

  vertices.push(0, y, 0);
  normals.push(0, y > 0 ? 1 : -1, 0);

  for (let i = 0; i <= segments; i++) {
    const theta = (i / segments) * Math.PI * 2;
    const x = Math.cos(theta) * radius;
    const z = Math.sin(theta) * radius;

    vertices.push(x, y, z);
    normals.push(0, y > 0 ? 1 : -1, 0);
  }

  return { vertices, normals };
}

const cylinder = generateCylinder(radius, height, segments);
const bottomDisk = generateDisk(radius, -height / 2, segments);
const fluid = generateCylinder(radius, fluidHeight, segments);

// Create buffers
function createBuffer(data) {
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
  return buffer;
}

const cylinderPositionBuffer = createBuffer(cylinder.vertices);
const cylinderNormalBuffer = createBuffer(cylinder.normals);
const bottomPositionBuffer = createBuffer(bottomDisk.vertices);
const bottomNormalBuffer = createBuffer(bottomDisk.normals);
const fluidPositionBuffer = createBuffer(fluid.vertices);
const fluidNormalBuffer = createBuffer(fluid.normals);

const aPositionLoc = gl.getAttribLocation(program, "a_Position");
const aNormalLoc = gl.getAttribLocation(program, "a_Normal");

gl.enableVertexAttribArray(aPositionLoc);
gl.enableVertexAttribArray(aNormalLoc);

// Uniform locations
const uModelMatrixLoc = gl.getUniformLocation(program, "u_ModelMatrix");
const uViewMatrixLoc = gl.getUniformLocation(program, "u_ViewMatrix");
const uProjectionMatrixLoc = gl.getUniformLocation(program, "u_ProjectionMatrix");
const uLightPositionLoc = gl.getUniformLocation(program, "u_LightPosition");
const uViewPositionLoc = gl.getUniformLocation(program, "u_ViewPosition");
const uColorLoc = gl.getUniformLocation(program, "u_Color");
const uTransparencyLoc = gl.getUniformLocation(program, "u_Transparency");

// Matrices
const modelMatrix = mat4.create();
const viewMatrix = mat4.create();
const projectionMatrix = mat4.create();
mat4.lookAt(viewMatrix, [0, 0.5, 1.5], [0, 0, 0], [0, 1, 0]);
mat4.perspective(projectionMatrix, Math.PI / 4, canvas.width / canvas.height, 0.1, 10.0);

// Mouse interaction
let lastX, lastY;
let rotX = 0, rotY = 0;

canvas.addEventListener("mousedown", (e) => {
  lastX = e.clientX;
  lastY = e.clientY;

  const mouseMoveHandler = (e) => {
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    rotY += dx * 0.01;
    rotX += dy * 0.01;

    lastX = e.clientX;
    lastY = e.clientY;
  };

  const mouseUpHandler = () => {
    canvas.removeEventListener("mousemove", mouseMoveHandler);
    canvas.removeEventListener("mouseup", mouseUpHandler);
  };

  canvas.addEventListener("mousemove", mouseMoveHandler);
  canvas.addEventListener("mouseup", mouseUpHandler);
});

// Render loop
function render() {
  gl.clearColor(1.0, 1.0, 1.0, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);

  mat4.identity(modelMatrix);
  mat4.rotateX(modelMatrix, modelMatrix, rotX);
  mat4.rotateY(modelMatrix, modelMatrix, rotY);

  gl.uniformMatrix4fv(uModelMatrixLoc, false, modelMatrix);
  gl.uniformMatrix4fv(uViewMatrixLoc, false, viewMatrix);
  gl.uniformMatrix4fv(uProjectionMatrixLoc, false, projectionMatrix);
  gl.uniform3f(uLightPositionLoc, 1.0, 1.0, 1.0);
  gl.uniform3f(uViewPositionLoc, 0, 0.5, 1.5);

  // Draw cylinder walls
  gl.uniform4f(uColorLoc, 0.0, 0.0, 0.0, 1.0); // Black
  gl.uniform1f(uTransparencyLoc, 1.0);
  gl.bindBuffer(gl.ARRAY_BUFFER, cylinderPositionBuffer);
  gl.vertexAttribPointer(aPositionLoc, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, cylinderNormalBuffer);
  gl.vertexAttribPointer(aNormalLoc, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, cylinder.vertices.length / 3);

  // Draw bottom disk
  gl.bindBuffer(gl.ARRAY_BUFFER, bottomPositionBuffer);
  gl.vertexAttribPointer(aPositionLoc, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, bottomNormalBuffer);
  gl.vertexAttribPointer(aNormalLoc, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.TRIANGLE_FAN, 0, bottomDisk.vertices.length / 3);

  // Draw fluid
  gl.uniform4f(uColorLoc, 0.0, 0.4, 1.0, 0.6); // Semi-transparent fluid
  gl.uniform1f(uTransparencyLoc, 0.6);
  gl.bindBuffer(gl.ARRAY_BUFFER, fluidPositionBuffer);
  gl.vertexAttribPointer(aPositionLoc, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, fluidNormalBuffer);
  gl.vertexAttribPointer(aNormalLoc, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, fluid.vertices.length / 3);

  requestAnimationFrame(render);
}

render();