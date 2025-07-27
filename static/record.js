"use strict";

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Recorder = function () {
  function Recorder(source, cfg) {
    var _this = this;

    _classCallCheck(this, Recorder);

    this.config = {
      bufferLen: 4096,
      mimeType: "audio/wav",
      numberOfChannels: 1,
      sampleRate: 16000
    };
    Object.assign(this.config, cfg);

    this.context = source.context;
    this.node = (this.context.createScriptProcessor || this.context.createJavaScriptNode).call(this.context, this.config.bufferLen, this.config.numberOfChannels, this.config.numberOfChannels);

    this.node.onaudioprocess = function (e) {
      if (!_this.recording) return;
      var buffer = [];
      for (var i = 0; i < _this.config.numberOfChannels; i++) {
        buffer.push(e.inputBuffer.getChannelData(i));
      }
      _this.worker.postMessage({
        command: "record",
        buffer: buffer
      });
    };

    source.connect(this.node);
    this.node.connect(this.context.destination);

    var worker = new Blob([_this.getWorkerCode()], { type: "text/javascript" });
    this.worker = new Worker(URL.createObjectURL(worker));
    this.worker.postMessage({
      command: "init",
      config: {
        sampleRate: this.context.sampleRate,
        numberOfChannels: this.config.numberOfChannels
      }
    });

    this.worker.onmessage = function (e) {
      var cb = _this.callbacks[e.data.command].pop();
      if (typeof cb === "function") {
        cb(e.data.data);
      }
    };

    this.callbacks = {
      getBuffer: [],
      exportWAV: []
    };
  }

  _createClass(Recorder, [{
    key: "getWorkerCode",
    value: function getWorkerCode() {
      var _this2 = this;

      var code = "\n      let recLength = 0,\n        recBuffers = [],\n        sampleRate,\n        numberOfChannels;\n\n      this.onmessage = function(e){\n        switch(e.data.command){\n          case 'init':\n            init(e.data.config);\n            break;\n          case 'record':\n            record(e.data.buffer);\n            break;\n          case 'exportWAV':\n            exportWAV(e.data.type);\n            break;\n          case 'getBuffer':\n            getBuffer();\n            break;\n          case 'clear':\n            clear();\n            break;\n        }\n      };\n\n      function init(config){\n        sampleRate = config.sampleRate;\n        numberOfChannels = config.numberOfChannels;\n        initBuffers();\n      }\n\n      function record(inputBuffer){\n        for (var i = 0; i < numberOfChannels; i++){\n          recBuffers[i].push(inputBuffer[i]);\n        }\n        recLength += inputBuffer[0].length;\n      }\n\n      function exportWAV(type){\n        let buffers = [];\n        for (let i = 0; i < numberOfChannels; i++){\n          buffers.push(mergeBuffers(recBuffers[i], recLength));\n        }\n        let interleaved;\n        if (numberOfChannels === 2){\n          interleaved = interleave(buffers[0], buffers[1]);\n        } else {\n          interleaved = buffers[0];\n        }\n        let dataview = encodeWAV(interleaved);\n        let audioBlob = new Blob([dataview], { type: type });\n\n        this.postMessage({ command: 'exportWAV', data: audioBlob });\n      }\n\n      function getBuffer(){\n        let buffers = [];\n        for (let i = 0; i < numberOfChannels; i++){\n          buffers.push(mergeBuffers(recBuffers[i], recLength));\n        }\n        this.postMessage({ command: 'getBuffer', data: buffers });\n      }\n\n      function clear(){\n        recLength = 0;\n        recBuffers = [];\n        initBuffers();\n      }\n\n      function initBuffers(){\n        for (let i = 0; i < numberOfChannels; i++){\n          recBuffers[i] = [];\n        }\n      }\n\n      function mergeBuffers(recBuffers, recLength){\n        let result = new Float32Array(recLength);\n        let offset = 0;\n        for (let i = 0; i < recBuffers.length; i++){\n          result.set(recBuffers[i], offset);\n          offset += recBuffers[i].length;\n        }\n        return result;\n      }\n\n      function interleave(inputL, inputR){\n        let length = inputL.length + inputR.length;\n        let result = new Float32Array(length);\n\n        let index = 0,\n          inputIndex = 0;\n\n        while (index < length){\n          result[index++] = inputL[inputIndex];\n          result[index++] = inputR[inputIndex];\n          inputIndex++;\n        }\n        return result;\n      }\n\n      function floatTo16BitPCM(output, offset, input){\n        for (let i = 0; i < input.length; i++, offset+=2){\n          let s = Math.max(-1, Math.min(1, input[i]));\n          output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);\n        }\n      }\n\n      function writeString(view, offset, string){\n        for (let i = 0; i < string.length; i++){\n          view.setUint8(offset + i, string.charCodeAt(i));\n        }\n      }\n\n      function encodeWAV(samples){\n        let buffer = new ArrayBuffer(44 + samples.length * 2);\n        let view = new DataView(buffer);\n\n        /* RIFF identifier */\n        writeString(view, 0, 'RIFF');\n        /* RIFF chunk length */\n        view.setUint32(4, 36 + samples.length * 2, true);\n        /* RIFF type */\n        writeString(view, 8, 'WAVE');\n        /* format chunk identifier */\n        writeString(view, 12, 'fmt ');\n        /* format chunk length */\n        view.setUint32(16, 16, true);\n        /* sample format (raw) */\n        view.setUint16(20, 1, true);\n        /* channel count */\n        view.setUint16(22, numberOfChannels, true);\n        /* sample rate */\n        view.setUint32(24, sampleRate, true);\n        /* byte rate (sample rate * block align) */\n        view.setUint32(28, sampleRate * (numberOfChannels * 2), true);\n        /* block align (channel count * bytes per sample) */\n        view.setUint16(32, numberOfChannels * 2, true);\n        /* bits per sample */\n        view.setUint16(34, 16, true);\n        /* data chunk identifier */\n        writeString(view, 36, 'data');\n        /* data chunk length */\n        view.setUint32(40, samples.length * 2, true);\n\n        floatTo16BitPCM(view, 44, samples);\n\n        return view;\n      }\n    ";
      return code;
    }
  }, {
    key: "record",
    value: function record() {
      this.recording = true;
    }
  }, {
    key: "stop",
    value: function stop() {
      this.recording = false;
    }
  }, {
    key: "clear",
    value: function clear() {
      this.worker.postMessage({ command: "clear" });
    }
  }, {
    key: "getBuffer",
    value: function getBuffer(cb) {
      cb = cb || this.config.callback;
      if (!cb) throw new Error("Callback not set");
      this.callbacks.getBuffer.push(cb);
      this.worker.postMessage({ command: "getBuffer" });
    }
  }, {
    key: "exportWAV",
    value: function exportWAV(cb, mimeType) {
      mimeType = mimeType || this.config.mimeType;
      cb = cb || this.config.callback;
      if (!cb) throw new Error("Callback not set");
      this.callbacks.exportWAV.push(cb);
      this.worker.postMessage({
        command: "exportWAV",
        type: mimeType
      });
    }
  }]);

  return Recorder;
}();

if (typeof module !== "undefined") {
  module.exports = Recorder;
}