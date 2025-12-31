
package com.k2fsa.sherpa.onnx

import android.content.ContentValues
import android.media.*
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.concurrent.thread

private const val TAG = "sherpa-onnx"

class MainActivity : AppCompatActivity() {

    private lateinit var recognizer: OnlineRecognizer
    private lateinit var textView: TextView
    private lateinit var startButton: Button

    private val sampleRate = 16000
    private val CHUNK_SECONDS = 30f

    data class SrtSeg(val start: Float, val end: Float, val text: String)
    private val segments = mutableListOf<SrtSeg>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        textView = findViewById(R.id.my_text)
        startButton = findViewById(R.id.record_button)

        initModel()

        startButton.setOnClickListener {
            textView.text = "Processing…"
            segments.clear()
            thread {
                processFile("input.mp3") // or input.wav
            }
        }
    }

    // ================= FILE → SRT =================

    private fun processFile(assetName: String) {
        val pcm = decodeToPcm(assetName)
        if (pcm.isEmpty()) {
            runOnUiThread { textView.text = "Decode failed" }
            return
        }

        val stream = recognizer.createStream()
        val chunkSamples = (sampleRate * CHUNK_SECONDS).toInt()

        var i = 0
        var lastText = ""
        var segStart = 0f

        while (i < pcm.size) {
            val end = minOf(i + chunkSamples, pcm.size)
            val chunk = pcm.copyOfRange(i, end)

            stream.acceptWaveform(chunk, sampleRate)

            while (recognizer.isReady(stream)) {
                recognizer.decode(stream)
            }

            val text = recognizer.getResult(stream).text
            val nowSec = end / sampleRate.toFloat()

            if (lastText.isBlank() && text.isNotBlank()) {
                segStart = i / sampleRate.toFloat()
            }
            if (text.isNotBlank()) lastText = text

            if (recognizer.isEndpoint(stream) || end == pcm.size) {
                if (lastText.isNotBlank()) {
                    segments.add(SrtSeg(segStart, nowSec, lastText))
                }
                recognizer.reset(stream)
                lastText = ""
            }

            i = end
        }

        stream.release()
        saveSrtDownloads()

        runOnUiThread {
            textView.text = "Saved to Downloads/output.srt"
        }
    }

    // ================= AUDIO DECODER =================

    private fun decodeToPcm(assetName: String): FloatArray {
        val extractor = MediaExtractor()
        val afd = assets.openFd(assetName)
        extractor.setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)

        var trackIndex = -1
        var format: MediaFormat? = null

        for (i in 0 until extractor.trackCount) {
            val f = extractor.getTrackFormat(i)
            if (f.getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true) {
                trackIndex = i
                format = f
                break
            }
        }

        if (trackIndex < 0 || format == null) return FloatArray(0)
        extractor.selectTrack(trackIndex)

        val codec = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
        codec.configure(format, null, null, 0)
        codec.start()

        val pcmList = ArrayList<Float>()
        val info = MediaCodec.BufferInfo()
        val buf = ByteArray(4096)

        while (true) {
            val inIdx = codec.dequeueInputBuffer(10000)
            if (inIdx >= 0) {
                val input = codec.getInputBuffer(inIdx)!!
                val size = extractor.readSampleData(input, 0)
                if (size < 0) {
                    codec.queueInputBuffer(
                        inIdx, 0, 0, 0,
                        MediaCodec.BUFFER_FLAG_END_OF_STREAM
                    )
                    break
                } else {
                    codec.queueInputBuffer(inIdx, 0, size, extractor.sampleTime, 0)
                    extractor.advance()
                }
            }

            val outIdx = codec.dequeueOutputBuffer(info, 10000)
            if (outIdx >= 0) {
                val outBuf = codec.getOutputBuffer(outIdx)!!
                outBuf.get(buf, 0, info.size)
                outBuf.clear()

                val bb = ByteBuffer.wrap(buf, 0, info.size)
                    .order(ByteOrder.LITTLE_ENDIAN)

                while (bb.remaining() >= 2) {
                    pcmList.add(bb.short / 32768f)
                }
                codec.releaseOutputBuffer(outIdx, false)
            }
        }

        codec.stop()
        codec.release()
        extractor.release()

        return pcmList.toFloatArray()
    }

    // ================= SAVE SRT ⇒ DOWNLOADS =================

    private fun saveSrtDownloads() {
        val resolver = contentResolver

        val values = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, "output.srt")
            put(MediaStore.Downloads.MIME_TYPE, "application/x-subrip")
            put(MediaStore.Downloads.IS_PENDING, 1)
        }

        val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values) ?: return
        resolver.openOutputStream(uri)?.use { os ->
            segments.forEachIndexed { i, s ->
                os.write("${i + 1}\n".toByteArray())
                os.write("${fmt(s.start)} --> ${fmt(s.end)}\n".toByteArray())
                os.write("${s.text.trim()}\n\n".toByteArray())
            }
        }

        values.clear()
        values.put(MediaStore.Downloads.IS_PENDING, 0)
        resolver.update(uri, values, null, null)
        Log.i(TAG, "SRT saved to Downloads")
    }

    private fun fmt(sec: Float): String {
        val ms = (sec * 1000).toInt()
        return "%02d:%02d:%02d,%03d".format(
            ms / 3600000,
            (ms % 3600000) / 60000,
            (ms % 60000) / 1000,
            ms % 1000
        )
    }

    // ================= MODEL INIT =================

    private fun initModel() {
        val config = OnlineRecognizerConfig(
            featConfig = getFeatureConfig(sampleRate, 80),
            modelConfig = getModelConfig(
                modelFile = "model.int8.onnx",   // here
                tokensFile = "tokens.txt",       // here
                type = -1                        // custom model
            )!!,
            endpointConfig = getEndpointConfig(),
            enableEndpoint = true
        )

        recognizer = OnlineRecognizer(assets, config)
    }
}
