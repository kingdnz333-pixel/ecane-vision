
import 'dart:async';
import 'dart:typed_data';

import 'package:battery_plus/battery_plus.dart';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

enum AlertLevel { safe, attention, danger }

class _DetectionResult {
  _DetectionResult({
    required this.label,
    required this.score,
    required this.box,
  });

  final String label;
  final double score;
  final Rect box;
}


class _PlaneSnapshot {
  _PlaneSnapshot({
    required this.bytes,
    required this.bytesPerRow,
    required this.bytesPerPixel,
  });

  final Uint8List bytes;
  final int bytesPerRow;
  final int? bytesPerPixel;
}

class _FrameSnapshot {
  _FrameSnapshot({
    required this.width,
    required this.height,
    required this.planes,
  });

  final int width;
  final int height;
  final List<_PlaneSnapshot> planes;

  factory _FrameSnapshot.fromCameraImage(CameraImage image) {
    return _FrameSnapshot(
      width: image.width,
      height: image.height,
      planes: image.planes
          .map(
            (plane) => _PlaneSnapshot(
              bytes: Uint8List.fromList(plane.bytes),
              bytesPerRow: plane.bytesPerRow,
              bytesPerPixel: plane.bytesPerPixel,
            ),
          )
          .toList(growable: false),
    );
  }
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.landscapeLeft,
    DeviceOrientation.landscapeRight,
  ]);

  runApp(const ECaneApp());
}

class ECaneApp extends StatelessWidget {
  const ECaneApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'E.CANE Vision',
      theme: ThemeData.dark(useMaterial3: true).copyWith(
        scaffoldBackgroundColor: Colors.black,
      ),
      home: const CameraScreen(),
    );
  }
}

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  final Battery _battery = Battery();
  final FlutterTts _tts = FlutterTts();

  Interpreter? _interpreter;
  bool _modelLoaded = false;
  bool _inputIsFloat32 = false;
  int _inputSize = 300;
  int _maxDetections = 10;

  CameraController? _cameraController;
  final List<CameraDescription> _availableCameras = [];

  StreamSubscription<BluetoothAdapterState>? _bluetoothSubscription;
  Timer? _batteryTimer;
  // Captura e inferência são feitas sem guardar CameraImage bruto,
  // para evitar buffer descartado pelo coletor de lixo.
  _FrameSnapshot? _latestFrame;

  bool _initializing = true;
  bool _cameraReady = false;
  bool _audioReady = false;
  bool _bluetoothOn = false;
  bool _esp32Connected = false;
  bool _detectionActive = false;

  int _batteryPercent = 0;
  BatteryState _batteryState = BatteryState.unknown;

  AlertLevel _alertLevel = AlertLevel.safe;
  String _statusText = 'Analisando ambiente...';
  String _detectedObject = 'Nenhum obstáculo detectado';

  bool _isProcessingFrame = false;
  DateTime _lastAnalysisAt = DateTime.fromMillisecondsSinceEpoch(0);
  DateTime _lastAnnouncementAt = DateTime.fromMillisecondsSinceEpoch(0);
  String _lastAnnouncementText = '';

  BluetoothDevice? _caneDevice;
  BluetoothCharacteristic? _caneCharacteristic;
  StreamSubscription<List<ScanResult>>? _bleScanSubscription;
  bool _isBleConnecting = false;
  String _caneConnectionText = 'Sem conexão';

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  Future<void> _bootstrap() async {
    await _loadBatteryStatus();
    await _initAudio();
    _listenBluetooth();
    await _loadModel();
    await _initCamera(preferredLens: CameraLensDirection.back);

    if (!mounted) return;
    setState(() {
      _initializing = false;
    });

    _batteryTimer = Timer.periodic(
      const Duration(seconds: 30),
      (_) => _loadBatteryStatus(),
    );
  }

  Future<void> _loadBatteryStatus() async {
    try {
      final level = await _battery.batteryLevel;
      final state = await _battery.batteryState;

      if (!mounted) return;
      setState(() {
        _batteryPercent = level;
        _batteryState = state;
      });
    } catch (_) {
      // Mantém os valores anteriores quando a plataforma não permite leitura.
    }
  }

  Future<void> _initAudio() async {
    try {
      await _tts.setLanguage('pt-BR');
      await _tts.setSpeechRate(0.46);
      await _tts.setPitch(1.0);
      await _tts.setVolume(1.0);
      await _tts.awaitSpeakCompletion(true);

      if (!mounted) return;
      setState(() {
        _audioReady = true;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _audioReady = false;
      });
    }
  }

  void _listenBluetooth() {
    try {
      _bluetoothSubscription = FlutterBluePlus.adapterState.listen((state) {
        if (!mounted) return;
        setState(() {
          _bluetoothOn = state == BluetoothAdapterState.on;
        });
      });
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _bluetoothOn = false;
      });
    }
  }

  Future<void> _loadModel() async {
    try {
      _interpreter?.close();

      final options = InterpreterOptions()..threads = 2;
      _interpreter = await Interpreter.fromAsset(
        'assets/models/detect.tflite',
        options: options,
      );

      final inputTensor = _interpreter!.getInputTensor(0);
      final inputShape = inputTensor.shape;

      if (inputShape.isNotEmpty) {
        if (inputShape.length >= 3) {
          _inputSize = inputShape[1];
        }
      }

      _inputIsFloat32 = inputTensor.type == TfLiteType.kTfLiteFloat32;

      try {
        final outputShape = _interpreter!.getOutputTensor(0).shape;
        if (outputShape.length >= 2 && outputShape[1] > 0) {
          _maxDetections = outputShape[1];
        }
      } catch (_) {
        _maxDetections = 10;
      }

      _interpreter!.allocateTensors();
      _modelLoaded = true;
      if (mounted) {
        setState(() {
          _statusText = 'Modelo carregado';
          _detectedObject = 'Aguardando leitura';
        });
      }
    } catch (e) {
      _modelLoaded = false;
      if (!mounted) return;
      setState(() {
        _statusText = 'Modelo indisponível';
        _detectedObject = 'Verifique assets/models/detect.tflite';
      });
      debugPrint('Erro ao carregar modelo: $e');
    }
  }

  Future<void> _initCamera({
    required CameraLensDirection preferredLens,
  }) async {
    try {
      if (kIsWeb) {
        throw UnsupportedError('Camera preview is not enabled on web.');
      }

      _latestFrame = null;

      final cameras = await availableCameras();
      _availableCameras
        ..clear()
        ..addAll(cameras);

      if (_availableCameras.isEmpty) {
        if (!mounted) return;
        setState(() {
          _cameraReady = false;
          _statusText = 'Nenhuma câmera encontrada';
          _detectedObject = 'Câmera indisponível';
        });
        return;
      }

      final selected = _selectCamera(preferredLens);
      await _cameraController?.dispose();

      final controller = CameraController(
        selected,
        ResolutionPreset.medium,
        enableAudio: false,
      );

      await controller.initialize();

      if (!mounted) {
        await controller.dispose();
        return;
      }

      setState(() {
        _cameraController = controller;
        _cameraReady = true;
        _statusText = 'Analisando ambiente...';
      });

      try {
        await controller.startImageStream(_onCameraFrame);
      } catch (_) {
        // Se o stream não iniciar, o preview continua e a IA entra depois.
      }
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _cameraReady = false;
        _statusText = 'Câmera indisponível nesta plataforma';
        _detectedObject = 'Abra em um tablet Android para testar a câmera';
      });
    }
  }

  CameraDescription _selectCamera(CameraLensDirection preferredLens) {
    final preferred = _availableCameras.where(
      (camera) => camera.lensDirection == preferredLens,
    );
    if (preferred.isNotEmpty) {
      return preferred.first;
    }
    return _availableCameras.first;
  }

  Future<void> _switchCamera() async {
    if (_availableCameras.isEmpty) return;

    final currentLens = _cameraController?.description.lensDirection;
    final nextLens = currentLens == CameraLensDirection.front
        ? CameraLensDirection.back
        : CameraLensDirection.front;

    if (!mounted) return;
    setState(() {
      _cameraReady = false;
      _statusText = 'Trocando câmera...';
    });

    try {
      if (_cameraController?.value.isStreamingImages == true) {
        await _cameraController?.stopImageStream();
      }
    } catch (_) {
      // Ignora se o stream ainda não estiver ativo.
    }

    _latestFrame = null;

    await _cameraController?.dispose();
    _cameraController = null;

    await _initCamera(preferredLens: nextLens);
  }


  bool _isTargetCaneResult(ScanResult result) {
    final advertisedName = result.advertisementData.advName.trim();
    final platformName = result.device.platformName.trim();
    final mergedName = '$advertisedName $platformName'.toUpperCase();

    final serviceUuids = result.advertisementData.serviceUuids
        .map((uuid) => uuid.toString().toUpperCase())
        .toList(growable: false);

    final matchesName = mergedName.contains('HM-10') ||
        mergedName.contains('HM10') ||
        mergedName.contains('KEYESTUDIO') ||
        mergedName.contains('HMSOFT') ||
        mergedName.contains('BT05');

    final matchesService = serviceUuids.any((uuid) => uuid.contains('FFE0'));

    return matchesName || matchesService;
  }

  BluetoothCharacteristic? _findWritableCharacteristic(
    List<BluetoothService> services,
  ) {
    for (final service in services) {
      final serviceUuid = service.serviceUuid.toString().toUpperCase();
      final isHmService = serviceUuid.contains('FFE0') || serviceUuid.contains('FFE1');

      if (isHmService) {
        for (final characteristic in service.characteristics) {
          final charUuid = characteristic.characteristicUuid.toString().toUpperCase();
          final canWrite = characteristic.properties.write ||
              characteristic.properties.writeWithoutResponse;

          if (charUuid.contains('FFE1') && canWrite) {
            return characteristic;
          }
        }
      }
    }

    for (final service in services) {
      for (final characteristic in service.characteristics) {
        if (characteristic.properties.write ||
            characteristic.properties.writeWithoutResponse) {
          return characteristic;
        }
      }
    }

    return null;
  }

  Future<void> _connectToCane() async {
    if (_isBleConnecting || _esp32Connected) return;

    _isBleConnecting = true;
    _bleScanSubscription?.cancel();
    _bleScanSubscription = null;

    if (mounted) {
      setState(() {
        _caneConnectionText = 'Procurando bengala...';
      });
    }

    final completer = Completer<BluetoothDevice?>();

    try {
      await FlutterBluePlus.stopScan();

      _bleScanSubscription = FlutterBluePlus.scanResults.listen((results) async {
        for (final result in results) {
          if (!_isTargetCaneResult(result)) continue;

          if (!completer.isCompleted) {
            completer.complete(result.device);
          }
          break;
        }
      });

      await FlutterBluePlus.startScan(
        timeout: const Duration(seconds: 8),
      );

      if (!completer.isCompleted) {
        completer.complete(null);
      }

      final device = await completer.future;
      if (device == null) {
        if (mounted) {
          setState(() {
            _esp32Connected = false;
            _caneConnectionText = 'Bengala não encontrada';
          });
        }
        return;
      }

      _caneDevice = device;
      _caneCharacteristic = null;

      try {
        await device.disconnect();
      } catch (_) {}

      await device.connect(
        timeout: const Duration(seconds: 15),
        autoConnect: false,
      );

      final services = await device.discoverServices();
      final characteristic = _findWritableCharacteristic(services);

      if (characteristic == null) {
        if (mounted) {
          setState(() {
            _esp32Connected = false;
            _caneConnectionText = 'Canal BLE não encontrado';
          });
        }
        return;
      }

      _caneCharacteristic = characteristic;

      if (mounted) {
        setState(() {
          _esp32Connected = true;
          _caneConnectionText = 'Bengala conectada';
        });
      }
    } catch (e) {
      debugPrint('Erro BLE: $e');
      if (mounted) {
        setState(() {
          _esp32Connected = false;
          _caneConnectionText = 'Falha na conexão';
        });
      }
    } finally {
      _isBleConnecting = false;
      await _bleScanSubscription?.cancel();
      _bleScanSubscription = null;
    }
  }

  Future<void> _disconnectFromCane() async {
    _bleScanSubscription?.cancel();
    _bleScanSubscription = null;

    try {
      await _caneCharacteristic?.setNotifyValue(false);
    } catch (_) {}

    try {
      await _caneDevice?.disconnect();
    } catch (_) {}

    _caneCharacteristic = null;
    _caneDevice = null;

    if (mounted) {
      setState(() {
        _esp32Connected = false;
        _caneConnectionText = 'Sem conexão';
      });
    }
  }

  void _onCameraFrame(CameraImage image) {
    if (!_cameraReady || _cameraController == null) return;
    if (!_modelLoaded || _interpreter == null) return;
    if (_isProcessingFrame) return;

    final now = DateTime.now();
    if (now.difference(_lastAnalysisAt).inMilliseconds < 650) return;

    _lastAnalysisAt = now;
    final snapshot = _FrameSnapshot.fromCameraImage(image);
    _isProcessingFrame = true;

    Future(() async {
      try {
        final input = _buildInputTensor(snapshot);
        final detections = _runInference(input);

        if (detections.isEmpty) {
          _applyVisionResult(
            level: AlertLevel.safe,
            objectLabel: 'Ambiente monitorado',
            status: 'Monitoramento ativo',
            speak: false,
            forceUpdate: false,
          );
        } else {
          final best = detections.first;
          final centerY = best.box.center.dy;

          final level = _classifyLevel(centerY, best.score, best.label);
          final status = switch (level) {
            AlertLevel.safe => 'Monitoramento ativo',
            AlertLevel.attention => 'Atenção: possível obstáculo',
            AlertLevel.danger => 'Cuidado, objeto na altura da cabeça!',
          };

          _applyVisionResult(
            level: level,
            objectLabel: '${best.label} • ${(best.score * 100).toStringAsFixed(0)}%',
            status: status,
            speak: level != AlertLevel.safe,
            forceUpdate: true,
          );
        }
      } catch (e) {
        debugPrint('Erro TFLite: $e');
        if (mounted) {
          setState(() {
            _detectionActive = false;
            _statusText = 'Leitura temporariamente indisponível';
            _detectedObject = 'Aguardando nova leitura';
          });
        }
      } finally {
        _isProcessingFrame = false;
      }
    });
  }

  Object _buildInputTensor(_FrameSnapshot image) {
    if (image.planes.length < 3) {
      if (_inputIsFloat32) {
        return Float32List(_inputSize * _inputSize * 3);
      }
      return Uint8List(_inputSize * _inputSize * 3);
    }

    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    final uvRowStride = uPlane.bytesPerRow;
    final uvPixelStride = uPlane.bytesPerPixel ?? vPlane.bytesPerPixel ?? 1;

    if (_inputIsFloat32) {
      final input = Float32List(_inputSize * _inputSize * 3);

      for (int y = 0; y < _inputSize; y++) {
        final py = (y * image.height ~/ _inputSize);

        for (int x = 0; x < _inputSize; x++) {
          final px = (x * image.width ~/ _inputSize);

          final yIndex = py * yPlane.bytesPerRow + px;
          final uvIndex = (py ~/ 2) * uvRowStride + (px ~/ 2) * uvPixelStride;

          final yVal = yPlane.bytes[yIndex];
          final uVal = uPlane.bytes[uvIndex];
          final vVal = vPlane.bytes[uvIndex];

          final r = (yVal + 1.402 * (vVal - 128)).clamp(0, 255).toDouble() / 255.0;
          final g = (yVal - 0.344136 * (uVal - 128) - 0.714136 * (vVal - 128))
                  .clamp(0, 255)
                  .toDouble() /
              255.0;
          final b = (yVal + 1.772 * (uVal - 128)).clamp(0, 255).toDouble() / 255.0;

          final index = (y * _inputSize * 3) + (x * 3);
          input[index] = r;
          input[index + 1] = g;
          input[index + 2] = b;
        }
      }

      return input;
    }

    final input = Uint8List(_inputSize * _inputSize * 3);

    for (int y = 0; y < _inputSize; y++) {
      final py = (y * image.height ~/ _inputSize);

      for (int x = 0; x < _inputSize; x++) {
        final px = (x * image.width ~/ _inputSize);

        final yIndex = py * yPlane.bytesPerRow + px;
        final uvIndex = (py ~/ 2) * uvRowStride + (px ~/ 2) * uvPixelStride;

        final yVal = yPlane.bytes[yIndex];
        final uVal = uPlane.bytes[uvIndex];
        final vVal = vPlane.bytes[uvIndex];

        final r = (yVal + 1.402 * (vVal - 128)).round().clamp(0, 255).toInt();
        final g = (yVal - 0.344136 * (uVal - 128) - 0.714136 * (vVal - 128))
            .round()
            .clamp(0, 255)
            .toInt();
        final b = (yVal + 1.772 * (uVal - 128)).round().clamp(0, 255).toInt();

        final index = (y * _inputSize * 3) + (x * 3);
        input[index] = r;
        input[index + 1] = g;
        input[index + 2] = b;
      }
    }

    return input;
  }

  List<_DetectionResult> _runInference(Object input) {
    final outputBoxes = List.generate(
      1,
      (_) => List.generate(_maxDetections, (_) => List.filled(4, 0.0)),
    );
    final outputClasses = List.generate(
      1,
      (_) => List.filled(_maxDetections, 0.0),
    );
    final outputScores = List.generate(
      1,
      (_) => List.filled(_maxDetections, 0.0),
    );
    final numDetections = List.filled(1, 0.0);

    final outputs = <int, Object>{
      0: outputBoxes,
      1: outputClasses,
      2: outputScores,
      3: numDetections,
    };

    final interpreter = _interpreter;
    if (interpreter == null) return <_DetectionResult>[];

    interpreter.runForMultipleInputs([input], outputs);

    final detections = <_DetectionResult>[];
    final boxes = outputBoxes[0];
    final classes = outputClasses[0];
    final scores = outputScores[0];

    for (int i = 0; i < scores.length; i++) {
      final score = scores[i].toDouble();
      if (score < 0.40) continue;

      final classIndex = classes[i].round();
      final box = boxes[i];

      final normalizedBox = Rect.fromLTRB(
        box[1].toDouble().clamp(0.0, 1.0),
        box[0].toDouble().clamp(0.0, 1.0),
        box[3].toDouble().clamp(0.0, 1.0),
        box[2].toDouble().clamp(0.0, 1.0),
      );

      detections.add(
        _DetectionResult(
          label: _labelForClassIndex(classIndex),
          score: score,
          box: normalizedBox,
        ),
      );
    }

    detections.sort((a, b) => b.score.compareTo(a.score));
    return detections;
  }

  AlertLevel _classifyLevel(double centerY, double score, String label) {
    final isPerson = label == 'Pessoa';

    if (score >= 0.60 && (centerY <= 0.42 || isPerson)) {
      return AlertLevel.danger;
    }
    if (score >= 0.45 && centerY <= 0.72) {
      return AlertLevel.attention;
    }
    return AlertLevel.safe;
  }

  String _labelForClassIndex(int index) {
    switch (index) {
      case 0:
        return 'Pessoa';
      case 1:
        return 'Bicicleta';
      case 2:
        return 'Carro';
      case 3:
        return 'Moto';
      case 5:
        return 'Ônibus';
      case 7:
        return 'Caminhão';
      case 10:
        return 'Semáforo';
      case 11:
        return 'Hidrante';
      case 13:
        return 'Banco';
      case 14:
        return 'Pássaro';
      case 16:
        return 'Cachorro';
      case 18:
        return 'Cavalo';
      case 19:
        return 'Ovelha';
      case 20:
        return 'Vaca';
      case 22:
        return 'Urso';
      case 24:
        return 'Mala';
      case 26:
        return 'Guarda-chuva';
      case 28:
        return 'Bolsa';
      case 31:
        return 'Frisbee';
      case 32:
        return 'Esqui';
      case 33:
        return 'Snowboard';
      case 34:
        return 'Bola';
      case 35:
        return 'Pipa';
      case 36:
        return 'Taco de baseball';
      case 37:
        return 'Luva';
      case 38:
        return 'Skate';
      case 39:
        return 'Prancha';
      case 41:
        return 'Garrafa';
      case 44:
        return 'Copo';
      case 46:
        return 'Banana';
      case 47:
        return 'Maçã';
      case 48:
        return 'Sanduíche';
      case 49:
        return 'Laranja';
      case 50:
        return 'Brócolis';
      case 51:
        return 'Cenoura';
      case 52:
        return 'Cachorro-quente';
      case 53:
        return 'Pizza';
      case 56:
        return 'Cadeira';
      case 57:
        return 'Sofá';
      case 58:
        return 'Mesa';
      case 59:
        return 'Vaso';
      case 60:
        return 'Cama';
      case 62:
        return 'TV';
      case 63:
        return 'Notebook';
      case 64:
        return 'Mouse';
      case 65:
        return 'Controle';
      case 67:
        return 'Celular';
      case 69:
        return 'Micro-ondas';
      case 70:
        return 'Forno';
      case 72:
        return 'Pia';
      case 73:
        return 'Geladeira';
      case 74:
        return 'Livro';
      default:
        return 'Objeto detectado';
    }
  }

  String _speechForLevel(AlertLevel level) {
    switch (level) {
      case AlertLevel.safe:
        return '';
      case AlertLevel.attention:
        return 'Objeto à altura da cabeça';
      case AlertLevel.danger:
        return 'Cuidado, objeto à altura da cabeça';
    }
  }

  Future<void> _sendToCane(AlertLevel level) async {
    if (!_bluetoothOn || !_esp32Connected || _caneCharacteristic == null) return;

    final code = switch (level) {
      AlertLevel.safe => 'S',
      AlertLevel.attention => 'A',
      AlertLevel.danger => 'P',
    };

    try {
      final payload = Uint8List.fromList(code.codeUnits);
      await _caneCharacteristic!.write(
        payload,
        withoutResponse: _caneCharacteristic!.properties.writeWithoutResponse,
      );
    } catch (e) {
      debugPrint('Erro BLE ao enviar comando: $e');
      if (mounted) {
        setState(() {
          _esp32Connected = false;
          _caneConnectionText = 'Sem conexão';
        });
      }
    }
  }

  Future<void> _speak(String text) async {
    if (!_audioReady) return;

    final now = DateTime.now();
    if (text == _lastAnnouncementText &&
        now.difference(_lastAnnouncementAt).inSeconds < 2) {
      return;
    }

    if (now.difference(_lastAnnouncementAt).inMilliseconds < 900) {
      return;
    }

    _lastAnnouncementText = text;
    _lastAnnouncementAt = now;

    try {
      await _tts.speak(text);
    } catch (_) {
      // Silêncio de fallback.
    }
  }

  void _applyVisionResult({
    required AlertLevel level,
    required String objectLabel,
    required String status,
    bool speak = false,
    bool forceUpdate = false,
  }) {
    final shouldUpdate = forceUpdate ||
        level != _alertLevel ||
        objectLabel != _detectedObject ||
        status != _statusText ||
        !_detectionActive;

    if (!shouldUpdate) return;

    setState(() {
      _alertLevel = level;
      _detectedObject = objectLabel;
      _statusText = status;
      _detectionActive = true;
    });

    _sendToCane(level);

    if (speak) {
      final spoken = _speechForLevel(level);
      if (spoken.isNotEmpty) {
        _speak(spoken);
      }
    }
  }

  void _setDetectionOffline() {
    setState(() {
      _detectionActive = false;
      _alertLevel = AlertLevel.safe;
      _detectedObject = 'Nenhum obstáculo detectado';
      if (_cameraReady) {
        _statusText = 'Analisando ambiente...';
      }
    });
  }

  String _audioLabel() => _audioReady ? 'Ativo' : 'Desativado';

  String _detectionLabel() => _detectionActive ? 'Ativa' : 'Offline';

  String _cameraLabel() {
    final lens = _cameraController?.description.lensDirection;
    if (lens == CameraLensDirection.front) return 'Frontal';
    return 'Traseira';
  }

  String _signalLabel() {
    if (!_bluetoothOn) return 'Bluetooth off';
    if (_isBleConnecting) return 'Conectando...';
    return _esp32Connected ? _caneConnectionText : 'Sem conexão';
  }

  Color _statusColor() {
    if (_cameraReady) return const Color(0xFF20D36B);
    return const Color(0xFFE74C3C);
  }

  Color _batteryColor() {
    if (_batteryState == BatteryState.charging) {
      return const Color(0xFF20D36B);
    }
    if (_batteryPercent >= 25) {
      return Colors.white70;
    }
    return const Color(0xFFF1C40F);
  }

  IconData _batteryIcon() {
    if (_batteryState == BatteryState.charging) {
      return Icons.battery_charging_full;
    }
    if (_batteryPercent >= 95) return Icons.battery_full;
    if (_batteryPercent >= 80) return Icons.battery_6_bar;
    if (_batteryPercent >= 65) return Icons.battery_5_bar;
    if (_batteryPercent >= 50) return Icons.battery_4_bar;
    if (_batteryPercent >= 35) return Icons.battery_3_bar;
    if (_batteryPercent >= 20) return Icons.battery_2_bar;
    if (_batteryPercent >= 5) return Icons.battery_1_bar;
    return Icons.battery_alert;
  }

  @override
  void dispose() {
    _batteryTimer?.cancel();
    _bluetoothSubscription?.cancel();
    _bleScanSubscription?.cancel();
    try {
      _cameraController?.stopImageStream();
    } catch (_) {}
    _cameraController?.dispose();
    _caneCharacteristic = null;
    _caneDevice = null;
    _interpreter?.close();
    _tts.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _cameraController;
    final hasCameraPreview = _cameraReady && controller != null;
    final topStatusLabel = _cameraReady ? 'Ativo' : 'Desativado';
    final topStatusColor = _statusColor();

    return Scaffold(
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            final w = constraints.maxWidth;
            final h = constraints.maxHeight;

            const topBarHeight = 60.0;
            const bottomBarHeight = 100.0;
            const sidePadding = 14.0;
            const bandGap = 8.0;

            final middleHeight = h - topBarHeight - bottomBarHeight - 20;
            final bandHeight = (middleHeight - bandGap * 2) / 3;

            final leftCardWidth = (w * 0.19).clamp(130.0, 170.0);
            final rightTextWidth = (w * 0.28).clamp(200.0, 320.0);

            return Stack(
              children: [
                Positioned.fill(
                  child: hasCameraPreview
                      ? Center(
                          child: AspectRatio(
                            aspectRatio: controller.value.aspectRatio,
                            child: CameraPreview(controller),
                          ),
                        )
                      : Container(color: Colors.black),
                ),

                Positioned.fill(
                  child: IgnorePointer(
                    child: Container(
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          begin: Alignment.topCenter,
                          end: Alignment.bottomCenter,
                          colors: [
                            Colors.black.withOpacity(0.16),
                            Colors.black.withOpacity(0.05),
                            Colors.black.withOpacity(0.20),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),

                Positioned(
                  left: 16,
                  right: 16,
                  top: 10,
                  child: Row(
                    children: [
                      const Text(
                        'E.CANE Vision',
                        style: TextStyle(
                          fontSize: 22,
                          fontWeight: FontWeight.w900,
                          letterSpacing: 0.2,
                        ),
                      ),
                      const SizedBox(width: 12),
                      _pill(
                        label: topStatusLabel,
                        color: topStatusColor,
                      ),
                      const Spacer(),
                      _topRightChip(
                        icon: _batteryIcon(),
                        label: '$_batteryPercent%',
                        iconColor: _batteryColor(),
                      ),
                      const SizedBox(width: 10),
                      _topRightChip(
                        icon: _bluetoothOn
                            ? Icons.bluetooth
                            : Icons.bluetooth_disabled,
                        label: _bluetoothOn ? 'BT on' : 'BT off',
                        iconColor: _bluetoothOn
                            ? const Color(0xFF20D36B)
                            : Colors.white70,
                      ),
                      const SizedBox(width: 10),
                      _iconButton(
                        icon: Icons.tune,
                        onTap: _openConfigSheet,
                      ),
                    ],
                  ),
                ),

                Positioned(
                  left: sidePadding,
                  right: sidePadding,
                  top: topBarHeight,
                  height: bandHeight,
                  child: _zoneBand(
                    color: const Color(0xFFE74C3C),
                    title: 'PERIGO',
                    subtitle: 'Altura da cabeça',
                    note: 'Detecta obstáculos acima da linha segura.',
                    active: _alertLevel == AlertLevel.danger,
                    leftCardWidth: leftCardWidth,
                    rightTextWidth: rightTextWidth,
                  ),
                ),

                Positioned(
                  left: sidePadding,
                  right: sidePadding,
                  top: topBarHeight + bandHeight + bandGap,
                  height: bandHeight,
                  child: _zoneBand(
                    color: const Color(0xFFF1C40F),
                    title: 'ATENÇÃO',
                    subtitle: 'Monitore o ambiente',
                    note: 'Obstáculos que pedem atenção imediata.',
                    active: _alertLevel == AlertLevel.attention,
                    leftCardWidth: leftCardWidth,
                    rightTextWidth: rightTextWidth,
                  ),
                ),

                Positioned(
                  left: sidePadding,
                  right: sidePadding,
                  top: topBarHeight + (bandHeight + bandGap) * 2,
                  height: bandHeight,
                  child: _zoneBand(
                    color: const Color(0xFF48D96E),
                    title: 'ANALISANDO',
                    subtitle: 'Leitura contínua',
                    note: 'Área de leitura segura enquanto não há alerta.',
                    active: _alertLevel == AlertLevel.safe,
                    leftCardWidth: leftCardWidth,
                    rightTextWidth: rightTextWidth,
                  ),
                ),

                Positioned(
                  left: 16,
                  right: 16,
                  bottom: 12,
                  child: Container(
                    height: 86,
                    padding: const EdgeInsets.symmetric(horizontal: 14),
                    decoration: BoxDecoration(
                      color: Colors.black.withOpacity(0.72),
                      borderRadius: BorderRadius.circular(22),
                      border: Border.all(color: Colors.white24),
                    ),
                    child: Row(
                      children: [
                        Expanded(
                          child: _bottomCell(
                            icon: Icons.volume_up_outlined,
                            label: 'Audio',
                            value: _audioLabel(),
                          ),
                        ),
                        _verticalDivider(),
                        Expanded(
                          child: _bottomCell(
                            icon: Icons.radar_outlined,
                            label: 'Detecção',
                            value: _detectionLabel(),
                          ),
                        ),
                        _verticalDivider(),
                        Expanded(
                          child: _bottomCell(
                            icon: Icons.photo_camera_outlined,
                            label: 'Camera',
                            value: _cameraLabel(),
                          ),
                        ),
                        _verticalDivider(),
                        Expanded(
                          child: _bottomCell(
                            icon: Icons.signal_cellular_alt,
                            label: 'Sinal',
                            value: _signalLabel(),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

                if (!hasCameraPreview)
                  Positioned.fill(
                    child: IgnorePointer(
                      child: Center(
                        child: Container(
                          width: 420,
                          padding: const EdgeInsets.all(18),
                          margin: const EdgeInsets.symmetric(horizontal: 24),
                          decoration: BoxDecoration(
                            color: Colors.black.withOpacity(0.50),
                            borderRadius: BorderRadius.circular(24),
                            border: Border.all(color: Colors.white24),
                          ),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const Icon(
                                Icons.videocam_off_outlined,
                                size: 34,
                                color: Colors.white70,
                              ),
                              const SizedBox(height: 10),
                              Text(
                                _statusText,
                                textAlign: TextAlign.center,
                                style: const TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.w800,
                                ),
                              ),
                              const SizedBox(height: 6),
                              Text(
                                _detectedObject,
                                textAlign: TextAlign.center,
                                style: const TextStyle(
                                  fontSize: 12,
                                  color: Colors.white70,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),

                if (_initializing && !hasCameraPreview)
                  const Positioned.fill(
                    child: Center(
                      child: CircularProgressIndicator(),
                    ),
                  ),
              ],
            );
          },
        ),
      ),
    );
  }

  Widget _zoneBand({
    required Color color,
    required String title,
    required String subtitle,
    required String note,
    required bool active,
    required double leftCardWidth,
    required double rightTextWidth,
  }) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 180),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(active ? 0.30 : 0.20),
        borderRadius: BorderRadius.circular(22),
        border: Border.all(
          color: active ? Colors.white : Colors.white24,
          width: active ? 2 : 1,
        ),
        boxShadow: active
            ? [
                BoxShadow(
                  color: color.withOpacity(0.14),
                  blurRadius: 16,
                  spreadRadius: 1,
                ),
              ]
            : null,
      ),
      child: Row(
        children: [
          SizedBox(
            width: leftCardWidth,
            child: _zoneCard(
              title: title,
              subtitle: subtitle,
              color: color,
            ),
          ),
          const SizedBox(width: 18),
          const Expanded(child: SizedBox.shrink()),
          SizedBox(
            width: rightTextWidth,
            child: Align(
              alignment: Alignment.centerRight,
              child: Text(
                note,
                textAlign: TextAlign.right,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                  fontSize: 14,
                  height: 1.0,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _zoneCard({
    required String title,
    required String subtitle,
    required Color color,
  }) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.95),
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: color.withOpacity(0.16),
            blurRadius: 12,
            spreadRadius: 1,
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            title,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w900,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            subtitle,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(
              fontSize: 10,
              height: 1.0,
              color: Colors.white,
            ),
          ),
        ],
      ),
    );
  }

  Widget _bottomCell({
    required IconData icon,
    required String label,
    required String value,
  }) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(icon, size: 18, color: Colors.white70),
        const SizedBox(height: 6),
        Text(
          label,
          style: const TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w700,
            color: Colors.white70,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          value,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w900,
            color: Colors.white,
          ),
        ),
      ],
    );
  }

  Widget _verticalDivider() {
    return Container(
      width: 1,
      height: 34,
      margin: const EdgeInsets.symmetric(horizontal: 8),
      color: Colors.white24,
    );
  }

  Widget _pill({
    required String label,
    required Color color,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.92),
        borderRadius: BorderRadius.circular(999),
      ),
      child: Text(
        label,
        style: const TextStyle(
          fontSize: 12,
          fontWeight: FontWeight.w800,
          color: Colors.white,
        ),
      ),
    );
  }

  Widget _topRightChip({
    required IconData icon,
    required String label,
    required Color iconColor,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.22),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.white24),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: iconColor),
          const SizedBox(width: 8),
          Text(
            label,
            style: const TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w700,
              color: Colors.white70,
            ),
          ),
        ],
      ),
    );
  }

  Widget _iconButton({
    required IconData icon,
    required VoidCallback onTap,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(999),
      child: Container(
        width: 38,
        height: 38,
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.22),
          borderRadius: BorderRadius.circular(999),
          border: Border.all(color: Colors.white24),
        ),
        child: Icon(icon, size: 18, color: Colors.white70),
      ),
    );
  }

  Future<void> _openConfigSheet() async {
    if (!mounted) return;

    await showModalBottomSheet<void>(
      context: context,
      backgroundColor: const Color(0xFF0D0D0D),
      showDragHandle: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (context) {
        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 18),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                _sheetAction(
                  icon: Icons.cameraswitch_outlined,
                  label: 'Trocar câmera',
                  onTap: () {
                    Navigator.pop(context);
                    _switchCamera();
                  },
                ),
                _sheetAction(
                  icon: Icons.volume_up_outlined,
                  label: 'Teste de áudio',
                  onTap: () {
                    Navigator.pop(context);
                    _speak('E.CANE Vision pronto');
                  },
                ),
                _sheetAction(
                  icon: Icons.bluetooth_searching_outlined,
                  label: _esp32Connected ? 'Desconectar bengala' : 'Conectar bengala',
                  onTap: () {
                    Navigator.pop(context);
                    if (_esp32Connected) {
                      _disconnectFromCane();
                    } else {
                      _connectToCane();
                    }
                  },
                ),
                _sheetAction(
                  icon: Icons.battery_std_outlined,
                  label: 'Atualizar bateria',
                  onTap: () {
                    Navigator.pop(context);
                    _loadBatteryStatus();
                  },
                ),
                _sheetAction(
                  icon: Icons.refresh,
                  label: 'Resetar alerta',
                  onTap: () {
                    Navigator.pop(context);
                    _setDetectionOffline();
                  },
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _sheetAction({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
  }) {
    return ListTile(
      leading: Icon(icon, color: Colors.white70),
      title: Text(
        label,
        style: const TextStyle(fontWeight: FontWeight.w700),
      ),
      onTap: onTap,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
    );
  }
}
