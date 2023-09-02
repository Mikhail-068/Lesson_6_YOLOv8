package com.example.yolov8

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.WindowManager
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import java.util.*
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    // объект для работы с потоком видео с камеры на экране
    private lateinit var previewView: PreviewView

    // объект для работы c кастомным view (RectView),
    // который отображает рамку и текст вокруг объектов
    private lateinit var rectView: RectView

    // объект инструмента, который необходим для работы с api-ONNX под JVM
    private lateinit var ortEnvironment: OrtEnvironment

    // Объект для работы с инструментом-оберткой моделей ONNX.
    // Позволяет нам работать с узлами модели
    private lateinit var session: OrtSession

    // объект класса для работы с данными
    // (подгружать модель НС, конвертировать форматы изображений и пр.)
    private val dataProcess = DataProcess(context = this)

    // Код активации разрешения (используется в функции, которая активирует разрешения)
    companion object {
        const val PERMISSION = 1
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        rectView = findViewById(R.id.rectView)

        // отключаем авто-выключение экрана
        // (по-умолчанию экран затемняется через некоторое время, мы это отключаем)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // вызываем функцию, которая проверит и, если оно не было запрошено ранее,
        // запросит у пользователя разрешение на работу с камерой устройства
        setPermissions()

        // загружаем и настраиваем модель
        load()

        // настраиваем и включаем камеру
        setCamera()
    }

    private fun setCamera() {
        // получаем объект для работы с камерой устройства
        val processCameraProvider = ProcessCameraProvider.getInstance(this).get()

        // устанавливаем вывод картинки с камеры на полный экран
        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER

        // необходимо выбрать камеру, с котрой будем работать (устанавливаем заднюю камеру)
        val cameraSelector =
            CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()

        // создаем объект для отображения картинки с камеры на экране
        // настраиваем формат, в котором будет отображаться картинка с камеры (устанавливаем 16:9)
        val preview = Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_16_9).build()

        // связываем созданный объект с объектом на нашем экране
        preview.setSurfaceProvider(previewView.surfaceProvider)

        // создаем объект, который будет заниматься анализом кадров с камеры устройства
        val analysis = ImageAnalysis.Builder().setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()

        // устанавливаем обработчик каждого кадра
        analysis.setAnalyzer(Executors.newSingleThreadExecutor()) {
            // вызываем обработку кадра
            imageProcess(it)
            // прекращаем работать с кадром
            it.close()
        }

        // привязываем процесс работы с камерой к жизненному циклу приложения,
        // это необходимо, например, чтобы работа камеры и процесс обработки изображения
        // были приостановлены, когда пользователь приостановил работу приложения (свернул его)
        processCameraProvider.bindToLifecycle(this, cameraSelector, preview, analysis)
    }

    private fun imageProcess(imageProxy: ImageProxy) {
        // преобразуем кадр в формат bitmap
        val bitmap = dataProcess.imageToBitmap(imageProxy)
        // преобразуем bitmap в floatBuffer для модели
        val floatBuffer = dataProcess.bitmapToFloatBuffer(bitmap)
        // ????
        val inputName = session.inputNames.iterator().next()
        //настраиваем входные данные для модели (в нашем случае [1 3 640 640])
        val shape = longArrayOf(
            DataProcess.BATCH_SIZE.toLong(),
            DataProcess.PIXEL_SIZE.toLong(),
            DataProcess.INPUT_SIZE.toLong(),
            DataProcess.INPUT_SIZE.toLong()
        )

        // создаем тензор на основе кадра и заданных параметров
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, floatBuffer, shape)

        // запускаем обработку данных и получаем результат
        val resultTensor = session.run(Collections.singletonMap(inputName, inputTensor))

        // берем первый элемент из массива результатов
        val outputs = resultTensor.get(0).value as Array<*> // [1 84 8400]

        // преобразуем выходные данные модели
        val results = dataProcess.outputsToNPMSPredictions(outputs)

        // отображаем данные на экране (рисуем рамку и подписи)
        rectView.transformRect(results)
        rectView.invalidate()
    }

    private fun load() {
        // подгружаем нашу модель
        dataProcess.loadModel()
        // подгружаем файл с сущностями для распознавания
        dataProcess.loadLabel()

        // получить объект для работы с ONNX
        ortEnvironment = OrtEnvironment.getEnvironment()
        // создаем сессию ONNX
        session = ortEnvironment.createSession(
            this.filesDir.absolutePath.toString() + "/" + DataProcess.FILE_NAME,
            OrtSession.SessionOptions()
        )

        // загружаем список распознаваемых сущностей во view,
        // который на кадрах рисует границы и подписи
        rectView.setClassLabel(dataProcess.classes)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        if (requestCode == PERMISSION) {
            // проверяем установку разрешений после их программной активации
            grantResults.forEach {
                // если они не установлены
                if (it != PackageManager.PERMISSION_GRANTED) {
                    // выводим предупреждение на экран
                    Toast.makeText(this, "Проверьте разрешения для приложения!", Toast.LENGTH_SHORT).show()
                    finish()
                }
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }

    private fun setPermissions() {
        // считываем в лист разрешение из манифеста приложения
        val permissions = ArrayList<String>()
        permissions.add(android.Manifest.permission.CAMERA)

        // двигаемся по всем элементам листа
        permissions.forEach {
            // проверяем, есть ли разрешение от пользователя на использование того, что мы запрашивали
            if (ActivityCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED) {
                // если нет, то запрашиваем разрешение у пользователя
                ActivityCompat.requestPermissions(this, permissions.toTypedArray(), PERMISSION)
            }
        }
    }
}