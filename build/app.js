/* Diện Chẩn AR - warps a phác đồ onto the live face.
 *
 * The heavy work happened offline: scripts/bake_web.py detected each chart's
 * landmarks, built the expansion ring, ran Delaunay, cropped the art to the
 * mesh and wrote meshes.json. Here the mesh is just a vertex buffer whose UVs
 * are fixed and whose positions are replaced every frame with the tracked
 * landmarks - one textured drawElements, so the GPU does the warp.
 */

import { FilesetResolver, FaceLandmarker } from './vendor/vision_bundle.mjs';

const ui = {
  chart: document.getElementById('chart'),
  opacity: document.getElementById('opacity'),
  snapshot: document.getElementById('snapshot'),
  stop: document.getElementById('stop'),
  canvas: document.getElementById('view'),
  message: document.getElementById('overlay-message'),
  status: document.getElementById('status'),
};

const params = new URLSearchParams(location.search);
const stillSource = params.get('source');   // debug: run against an image
const confidence = Number(params.get('conf') ?? 0.5);

function setMessage(text) {
  ui.message.textContent = text ?? '';
  ui.message.classList.toggle('hidden', !text);
}

function setStatus(text) {
  ui.status.textContent = text;
}

/* ---------------------------------------------------------------- WebGL --- */

const VIDEO_VS = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main() {
  v_uv = vec2((a_pos.x + 1.0) * 0.5, (1.0 - a_pos.y) * 0.5);
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const VIDEO_FS = `
precision mediump float;
varying vec2 v_uv;
uniform sampler2D u_tex;
void main() { gl_FragColor = texture2D(u_tex, v_uv); }`;

const MESH_VS = `
attribute vec2 a_pos;   // landmark, normalised to the frame
attribute vec2 a_uv;    // fixed, baked
varying vec2 v_uv;
void main() {
  v_uv = a_uv;
  gl_Position = vec4(a_pos.x * 2.0 - 1.0, 1.0 - a_pos.y * 2.0, 0.0, 1.0);
}`;

const MESH_FS = `
precision mediump float;
varying vec2 v_uv;
uniform sampler2D u_tex;
uniform float u_opacity;
void main() {
  vec4 c = texture2D(u_tex, v_uv);
  gl_FragColor = vec4(c.rgb, c.a * u_opacity);
}`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader));
  }
  return shader;
}

function link(gl, vertexSource, fragmentSource) {
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }
  return program;
}

function makeTexture(gl) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  return texture;
}

class Renderer {
  constructor(canvas) {
    // preserveDrawingBuffer keeps the frame readable after compositing, which
    // is what makes canvas.toBlob() in snapshot() return the render instead of
    // a blank image.
    const options = { alpha: false, antialias: true, preserveDrawingBuffer: true };
    const gl = canvas.getContext('webgl2', options)
      || canvas.getContext('webgl', options);
    if (!gl) throw new Error('WebGL không khả dụng trên trình duyệt này.');
    this.gl = gl;
    this.canvas = canvas;

    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    this.videoProgram = link(gl, VIDEO_VS, VIDEO_FS);
    this.meshProgram = link(gl, MESH_VS, MESH_FS);

    this.quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);

    this.videoTexture = makeTexture(gl);
    this.chartTexture = makeTexture(gl);
    this.positions = gl.createBuffer();
    this.uvs = gl.createBuffer();
    this.indices = gl.createBuffer();
    this.indexCount = 0;
  }

  setChart(chart, image) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.chartTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.uvs);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(chart.uv.flat()), gl.STATIC_DRAW);

    const indices = new Uint16Array(chart.triangles.flat());
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indices);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    this.indexCount = indices.length;
  }

  resize(width, height) {
    if (this.canvas.width === width && this.canvas.height === height) return;
    this.canvas.width = width;
    this.canvas.height = height;
    this.gl.viewport(0, 0, width, height);
  }

  draw(frame, meshPositions, opacity) {
    const gl = this.gl;
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.videoProgram);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, frame);
    const quadAttribute = gl.getAttribLocation(this.videoProgram, 'a_pos');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.enableVertexAttribArray(quadAttribute);
    gl.vertexAttribPointer(quadAttribute, 2, gl.FLOAT, false, 0, 0);
    gl.uniform1i(gl.getUniformLocation(this.videoProgram, 'u_tex'), 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    if (!meshPositions || !this.indexCount) return;

    gl.useProgram(this.meshProgram);
    gl.bindTexture(gl.TEXTURE_2D, this.chartTexture);
    gl.uniform1i(gl.getUniformLocation(this.meshProgram, 'u_tex'), 0);
    gl.uniform1f(gl.getUniformLocation(this.meshProgram, 'u_opacity'), opacity);

    const positionAttribute = gl.getAttribLocation(this.meshProgram, 'a_pos');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positions);
    gl.bufferData(gl.ARRAY_BUFFER, meshPositions, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(positionAttribute);
    gl.vertexAttribPointer(positionAttribute, 2, gl.FLOAT, false, 0, 0);

    const uvAttribute = gl.getAttribLocation(this.meshProgram, 'a_uv');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.uvs);
    gl.enableVertexAttribArray(uvAttribute);
    gl.vertexAttribPointer(uvAttribute, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indices);
    gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_SHORT, 0);
  }
}

/* ------------------------------------------------------------------ mesh --- */

/* Mirrors dienchan.triangulation.with_expansion_ring. The ring covers forehead
 * and ears, which the 478-point mesh does not reach; it barely grows downward
 * because nothing useful lives below the chin. Must match the bake exactly or
 * the baked UVs land on the wrong vertices. */
function buildPositions(landmarks, hull, expansion, downScale, out) {
  let cx = 0, cy = 0;
  for (const point of landmarks) { cx += point.x; cy += point.y; }
  cx /= landmarks.length;
  cy /= landmarks.length;

  for (let i = 0; i < landmarks.length; i++) {
    out[i * 2] = landmarks[i].x;
    out[i * 2 + 1] = landmarks[i].y;
  }
  for (let k = 0; k < hull.length; k++) {
    const point = landmarks[hull[k]];
    const dx = point.x - cx;
    const dy = point.y - cy;
    const scale = dy > 0 ? downScale : expansion;
    const at = (landmarks.length + k) * 2;
    out[at] = cx + dx * scale;
    out[at + 1] = cy + dy * scale;
  }
  return out;
}

/* ------------------------------------------------------------------- app --- */

function loadImage(source) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Không tải được ${source}`));
    image.src = source;
  });
}

class App {
  constructor() {
    this.renderer = new Renderer(ui.canvas);
    this.manifest = null;
    this.chart = null;
    this.positions = null;
    this.landmarker = null;
    this.frameSource = null;
    this.stream = null;
    this.running = false;
    this.fps = 0;
    this.lastFrame = 0;
    this.timestamp = 0;
  }

  async start() {
    setMessage('Đang tải phác đồ…');
    this.manifest = await (await fetch('meshes.json')).json();
    ui.chart.replaceChildren();
    for (const chart of this.manifest.charts) {
      const option = document.createElement('option');
      option.value = chart.id;
      option.textContent = chart.label;
      ui.chart.append(option);
    }
    ui.chart.disabled = false;
    ui.chart.onchange = () => this.selectChart(ui.chart.value);

    setMessage('Đang tải mô hình nhận diện…');
    const fileset = await FilesetResolver.forVisionTasks('./vendor/wasm');
    this.landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: './vendor/face_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      // 0.5 is right for a real face. Drawings need it far lower, hence the
      // override - the same confidence sweep the Python side does on charts.
      minFaceDetectionConfidence: confidence,
      minTrackingConfidence: confidence,
    });

    // Browsers restore a <select>'s previous value across reloads, and they do
    // it after the options exist. Follow the control rather than assuming the
    // first chart, or the picker and the render disagree on startup.
    const chosen = this.manifest.charts.some((c) => c.id === ui.chart.value)
      ? ui.chart.value : this.manifest.charts[0].id;
    ui.chart.value = chosen;
    await this.selectChart(chosen);
    await this.openSource();

    ui.stop.disabled = false;
    ui.snapshot.disabled = false;
    ui.stop.onclick = () => this.stop();
    ui.snapshot.onclick = () => this.snapshot();

    setMessage(null);
    this.running = true;
    this.lastFrame = performance.now();
    requestAnimationFrame(() => this.loop());
  }

  async selectChart(id) {
    const chart = this.manifest.charts.find((c) => c.id === id);
    const image = await loadImage(chart.image);
    this.renderer.setChart(chart, image);
    this.chart = chart;
    this.positions = new Float32Array(
      (chart.landmarkCount + chart.hull.length) * 2);
  }

  async openSource() {
    if (stillSource) {
      // Debug path: a still image instead of the camera, so the pipeline can
      // be exercised without a webcam.
      this.frameSource = await loadImage(stillSource);
      ui.canvas.classList.add('no-mirror');
      this.renderer.resize(this.frameSource.naturalWidth,
                           this.frameSource.naturalHeight);
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Trình duyệt không cho phép truy cập camera '
        + '(cần HTTPS hoặc localhost).');
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    const video = document.createElement('video');
    video.srcObject = this.stream;
    video.playsInline = true;
    video.muted = true;
    await video.play();
    this.frameSource = video;
    this.renderer.resize(video.videoWidth, video.videoHeight);
  }

  loop() {
    if (!this.running) return;

    const source = this.frameSource;
    const width = source.videoWidth || source.naturalWidth;
    const height = source.videoHeight || source.naturalHeight;
    if (width && height) this.renderer.resize(width, height);

    this.timestamp += 33;
    let landmarks = null;
    try {
      const result = this.landmarker.detectForVideo(source, this.timestamp);
      landmarks = result.faceLandmarks?.[0] ?? null;
    } catch (error) {
      setStatus(`lỗi nhận diện: ${error.message}`);
    }

    let positions = null;
    if (landmarks && landmarks.length === this.chart.landmarkCount) {
      positions = buildPositions(landmarks, this.chart.hull,
        this.manifest.expansion, this.manifest.downScale, this.positions);
    }
    this.renderer.draw(source, positions, Number(ui.opacity.value));

    const now = performance.now();
    this.fps = 0.9 * this.fps + 0.1 * (1000 / Math.max(now - this.lastFrame, 1));
    this.lastFrame = now;
    setStatus(`${this.fps.toFixed(0)} fps · `
      + `${landmarks ? 'đã bám khuôn mặt' : 'không thấy khuôn mặt'} · `
      + `${this.chart.label}`);

    requestAnimationFrame(() => this.loop());
  }

  snapshot() {
    ui.canvas.toBlob((blob) => {
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = `dienchan-${this.chart.id}-${Date.now()}.png`;
      link.click();
      URL.revokeObjectURL(link.href);
    });
  }

  stop() {
    this.running = false;
    this.stream?.getTracks().forEach((track) => track.stop());
    if (this.frameSource?.srcObject) this.frameSource.srcObject = null;
    ui.stop.disabled = true;
    ui.snapshot.disabled = true;
    ui.chart.disabled = true;
    setMessage('Đã dừng. Tải lại trang để chạy lại.');
    setStatus('đã dừng');
  }
}

const app = new App();
window.__dienchan = app;   // handle for automated checks
app.start().catch((error) => {
  console.error(error);
  setMessage(error.message);
  setStatus(`lỗi: ${error.message}`);
});
