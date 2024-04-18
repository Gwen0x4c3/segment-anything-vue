<template>
  <div class="segment-container">
    <ElScrollbar class="tool-box">
      <div class="image-section">
        <div class="title">
          <div style="padding-left:15px">
            <el-icon><Picture></Picture></el-icon>
            <el-upload
              ref="uploadRef"
              style="display: inline-block;"
              :action="uploadUrl"
              :auto-upload="true"
              :show-file-list="false"
              :on-success="showImage"
              :on-error="handleUploadError"
            >
              <template #trigger>
                <span style="font-size: 18px;font-weight: 550;cursor: pointer;"
                  @click="selectFile"
                >选择图像</span>
              </template>
            </el-upload>
            <el-icon class="header-icon"></el-icon>
          </div>
        </div>
        <ElScrollbar height="350px">
          <div v-if="cutOuts.length === 0">
            <p>未进行抠图</p>
            <p>左键设置区域为前景</p>
            <p>右键设置区域为背景</p>
          </div>
          <img v-for="src in cutOuts" :src="src" alt="加载中"
               @click="openInNewTab(src)"/>
        </ElScrollbar>
      </div>
      <div class="options-section">
        <span class="option" @click="reset">重置</span>
        <span :class="'option'+(clicks.length===0?' disabled':'')" @click="undo">撤销</span>
        <span :class="'option'+(clickHistory.length===0?' disabled':'')" @click="redo">恢复</span>
      </div>
      <button :class="'segmentation-button'+(lock||clicks.length===0?' disabled':'')"
              @click="cutImage">分割</button>
      <button :class="'segmentation-button'+(lock||isEverything?' disabled':'')"
              @click="segmentEverything">分割所有</button>
    </ElScrollbar>
    <div class="segment-box">
      <div class="segment-wrapper" :style="{'left': left + 'px'}">
        <img v-show="path" id="segment-image" :src="url" :style="{width:w, height:h}" alt="加载失败" crossorigin="anonymous"
             @mousedown="handleMouseDown" @mouseenter="canvasVisible = true"
             @mouseout="() => {if (!this.clicks.length&&!this.isEverything) this.canvasVisible = false}"/>
        <canvas v-show="path && canvasVisible" id="segment-canvas" :width="originalSize.w" :height="originalSize.h"></canvas>
        <div id="point-box" :style="{width:w, height:h}"></div>
      </div>
    </div>
  </div>
</template>

<script>
import throttle from "@/util/throttle";
import LZString from "lz-string";
import {
  rleFrString,
  decodeRleCounts,
  decodeEverythingMask,
  getUniqueColor,
  cutOutImage,
  cutOutImageWithMaskColor, cutOutImageWithCategory
} from "@/util/mask_utils";
import {ElCollapse, ElCollapseItem, ElScrollbar} from "element-plus";
import {Picture} from '@element-plus/icons-vue'
export default {
  name: "Segment",
  components: {
    ElCollapse, ElCollapseItem, ElScrollbar, Picture
  },
  data() {
    return {
      uploadUrl: "http://localhost:8006/upload",
      image: null,
      clicks: [],
      clickHistory: [],
      originalSize: {w: 0, h: 0},
      w: 0,
      h: 0,
      left: 0,
      scale: 1,
      url: null,
      path: null,
      loading: false,
      lock: false,
      canvasVisible: true,
      cutOuts: [],
      isEverything: false
    }
  },
  mounted() {
    this.init()
  },
  methods: {
    async init() {
    },
    handleUploadError() {
      alert("上传失败");
      console.error("Error", arguments)
    },
    showImage(res) {
      console.log("上传成功", res);
      this.loadImage(res.path, res.src)
    },
    loadImage(path, url) {
      let image = new Image();
      image.src = url;
      image.onload = () => {
        let w = image.width, h = image.height
        let nw, nh
        let body = document.querySelector('.segment-box')
        let mw = body.clientWidth, mh = body.clientHeight
        let ratio = w / h
        if (ratio * mh > mw) {
          nw = mw
          nh = mw / ratio
        } else {
          nh = mh
          nw = ratio * mh
        }
        this.originalSize = {w, h}
        nw = parseInt(nw)
        nh = parseInt(nh)
        this.w = nw + 'px'
        this.h = nh + 'px'
        this.left = (mw - nw) / 2
        this.scale = nw / w
        this.url = url
        this.path = path
        console.log((this.scale > 1 ? '放大' : '缩小') + w + ' --> ' + nw)
        const img = document.getElementById('segment-image')
        img.addEventListener('contextmenu', e => e.preventDefault())
        img.addEventListener('mousemove', throttle(this.handleMouseMove, 150))
        const canvas = document.getElementById('segment-canvas')
        canvas.style.transform = `scale(${this.scale})`
      }
    },
    getClick(e) {
      let click = {
        x: e.offsetX,
        y: e.offsetY,
      }
      const imageScale = this.scale
      click.x /= imageScale;
      click.y /= imageScale;
      if(e.which === 3){ // 右键
        click.clickType = 0
      } else if(e.which === 1 || e.which === 0) { // 左键
        click.clickType = 1
      }
      return click
    },
    handleMouseMove(e) {
      if (this.isEverything) { // 分割所有模式，返回
        return;
      }
      if (this.clicks.length !== 0) { // 选择了点
        return;
      }
      if (this.lock) {
        return;
      }
      this.lock = true;
      let click = this.getClick(e);
      requestIdleCallback(() => {
        this.getMask([click])
      })
    },
    handleMouseDown(e) {
      e.preventDefault();
      e.stopPropagation();
      if (e.button === 1) {
        return;
      }
      // 如果是“分割所有”模式，返回
      if (this.isEverything) {
        return;
      }
      if (this.lock) {
        return;
      }
      this.lock = true
      let click = this.getClick(e);
      this.placePoint(e.offsetX, e.offsetY, click.clickType)
      this.clicks.push(click);
      requestIdleCallback(() => {
        this.getMask()
      })
    },
    placePoint(x, y, clickType) {
      let box = document.getElementById('point-box')
      let point = document.createElement('div')
      point.className = 'segment-point' + (clickType ? '' : ' negative')
      point.style = `position: absolute;
                      width: 10px;
                      height: 10px;
                      border-radius: 50%;
                      background-color: ${clickType?'#409EFF':'#F56C6C '};
                      left: ${x-5}px;
                      top: ${y-5}px`
      // 点的id是在clicks数组中的下标索引
      point.id = 'point-' + this.clicks.length
      box.appendChild(point)
    },
    removePoint(i) {
      const selector = 'point-' + i
      let point = document.getElementById(selector)
      if (point != null) {
        point.remove()
      }
    },
    getMask(clicks) {
      // 如果clicks为空，则是mouse move产生的click
      if (clicks == null) {
        clicks = this.clicks
      }
      const data = {
        path: this.path,
        clicks: clicks
      }
      console.log(data)
      this.$http.post('http://localhost:8006/segment', data, {
        headers: {
          "Content-Type": "application/json"
        }
      }).then(res => {
        const shape = res.shape
        const maskenc = LZString.decompressFromEncodedURIComponent(res.mask);
        const decoded = rleFrString(maskenc)
        this.drawCanvas(shape, decodeRleCounts(shape, decoded))
        this.lock = false
      }).catch(err => {
        console.error(err)
        this.$message.error("生成失败")
        this.lock = false
      })
    },
    segmentEverything() {
      if (this.isEverything) { // 上一次刚点过了
        return;
      }
      if (this.lock) {
        return;
      }
      this.lock = true
      this.reset()
      this.isEverything = true
      this.canvasVisible = true
      this.$http.get("http://localhost:8006/everything?path=" + this.path).then(res => {
        const shape = res.shape
        const counts = res.mask
        this.drawEverythingCanvas(shape, decodeEverythingMask(shape, counts))
      }).catch(err => {
        console.error(err)
        this.$message.error("生成失败")
      })
    },
    drawCanvas(shape, arr) {
      let height = shape[0],
          width = shape[1]
      console.log("height: ", height, " width: ", width)
      let canvas = document.getElementById('segment-canvas'),
          canvasCtx = canvas.getContext("2d"),
          imgData = canvasCtx.getImageData(0, 0, width, height),
          pixelData = imgData.data
      let i = 0
      for(let x = 0; x < width; x++){
        for(let y = 0; y < height; y++){
          if (arr[i++] === 0) { // 如果是0，是背景，遮住
            pixelData[0 + (width * y + x) * 4] = 40;
            pixelData[1 + (width * y + x) * 4] = 40;
            pixelData[2 + (width * y + x) * 4] = 40;
            pixelData[3 + (width * y + x) * 4] = 190;
          } else {
            pixelData[3 + (width * y + x) * 4] = 0;
          }
        }
      }
      canvasCtx.putImageData(imgData, 0, 0)
    },
    drawEverythingCanvas(shape, arr) {
      const height = shape[0],
          width = shape[1]
      console.log("height: ", height, " width: ", width)
      let canvas = document.getElementById('segment-canvas'),
          canvasCtx = canvas.getContext("2d"),
          imgData = canvasCtx.getImageData(0, 0, width, height),
          pixelData = imgData.data;
      const colorMap = {}
      let i = 0
      for(let y = 0; y < height; y++){
        for(let x = 0; x < width; x++){
          const category = arr[i++]
          const color = getUniqueColor(category, colorMap)
          pixelData[0 + (width * y + x) * 4] = color.r;
          pixelData[1 + (width * y + x) * 4] = color.g;
          pixelData[2 + (width * y + x) * 4] = color.b;
          pixelData[3 + (width * y + x) * 4] = 150;
        }
      }
      // 显示在图片上
      canvasCtx.putImageData(imgData, 0, 0)
      // 开始分割每一个mask的图片
      const image = document.getElementById('segment-image')
      Object.keys(colorMap).forEach(category => {
        cutOutImageWithCategory(this.originalSize, image, arr, category, blob => {
          const url = URL.createObjectURL(blob);
          this.cutOuts = [url, ...this.cutOuts]
        })
      })
    },
    reset() {
      for (let i = 0; i < this.clicks.length; i++) {
        this.removePoint(i)
      }
      this.clicks = []
      this.clickHistory = []
      this.isEverything = false
      this.clearCanvas()
    },
    undo() {
      if (this.clicks.length === 0)
        return
      const idx = this.clicks.length - 1
      const click = this.clicks[idx]
      this.clickHistory.push(click)
      this.clicks.splice(idx, 1)
      this.removePoint(idx)
      if (this.clicks.length) {
        this.getMask()
      } else {
        this.clearCanvas()
      }
    },
    redo() {
      if (this.clickHistory.length === 0)
        return
      const idx = this.clickHistory.length - 1
      const click = this.clickHistory[idx]
      console.log(this.clicks, this.clickHistory, click)
      this.placePoint(click.x * this.scale, click.y * this.scale, click.clickType)
      this.clicks.push(click)
      this.clickHistory.splice(idx, 1)
      this.getMask()
    },
    clearCanvas() {
      let canvas = document.getElementById('segment-canvas')
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
    },
    cutImage() {
      if (this.lock || this.clicks.length === 0) {
        return;
      }
      const canvas = document.getElementById('segment-canvas'),
          image = document.getElementById('segment-image')
      const {w, h} = this.originalSize
      cutOutImage(this.originalSize, image, canvas, blob => {
        const url = URL.createObjectURL(blob);
        this.cutOuts = [url, ...this.cutOuts]
        // 不需要之后用下面的清除文件
        // URL.revokeObjectURL(url);
      })
    },
    openInNewTab(src) {
      window.open(src, '_blank')
    }
  }
}
</script>

<style scoped lang="scss">
.segment-container {
  position: relative;
  padding-top: 10px;
}

.tool-box {
  position: absolute;
  left: 20px;
  top: 20px;
  width: 200px;
  height: 600px;
  border-radius: 20px;
  //background: pink;
  overflow: auto;
  box-shadow: 0 0 5px rgb(150, 150, 150);
  box-sizing: border-box;
  padding: 10px;

  .image-section {
    height: fit-content;
    width: 100%;
    .title {
      height: 48px;
      line-height: 48px;
      border-bottom: 1px solid lightgray;
      margin-bottom: 15px;
    }
  }

  .image-section img {
    max-width: 85%;
    max-height: 140px;
    margin: 10px auto;
    padding: 10px;
    box-sizing: border-box;
    object-fit: contain;
    display: block;
    transition: .3s;
    cursor: pointer;
  }
  .image-section img:hover {
    background: rgba(0, 30, 160, 0.3);
  }

  .image-section p {
    text-align: center;
  }

  .options-section {
    margin-top: 5px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px;
    box-sizing: border-box;
    border: 3px solid lightgray;
    border-radius: 20px;
  }
  .options-section:hover {
    border: 3px solid #59ACFF;
  }

  .option {
    font-size: 15px;
    padding: 5px 10px;
    cursor: pointer;
  }
  .option:hover {
    color: #59ACFF;
  }
  .option.disabled {
    color: gray;
    cursor: not-allowed;
  }

  .segmentation-button {
    margin-top: 5px;
    width: 100%;
    height: 40px;
    background-color: white;
    color: rgb(40, 40, 40);
    font-size: 17px;
    cursor: pointer;
    border: 3px solid lightgray;
    border-radius: 20px;
  }
  .segmentation-button:hover {
    border: 3px solid #59ACFF;
  }
  .segmentation-button.disabled {
    color: lightgray;
    cursor: not-allowed;
  }
}

.segment-box {
  position: relative;
  margin-left: calc(220px);
  width: calc(100% - 220px);
  height: calc(100vh - 80px);
  //background: #42b983;
  .segment-wrapper {
    position: absolute;
    left: 0;
    top: 0;
  }
  #segment-canvas {
    position: absolute;
    left: 0;
    top: 0;
    pointer-events: none;
    transform-origin: left top;
    z-index: 1;
  }
  #point-box {
    position: absolute;
    left: 0;
    top: 0;
    z-index: 2;
    pointer-events: none;
  }
  .segment-point {
    position: absolute;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background-color: #409EFF;
  }
  .segment-point.negative {
    background-color: #F56C6C;
  }
}
</style>