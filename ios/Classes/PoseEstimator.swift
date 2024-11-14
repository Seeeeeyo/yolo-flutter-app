import Foundation
import UIKit
import Vision

public class PoseEstimator: Predictor {
    private var poseModel: VNCoreMLModel!
    private var visionRequest: VNCoreMLRequest?
    private var currentBuffer: CVPixelBuffer?
    private var currentOnResultsListener: ResultsListener?
    private var currentOnInferenceTimeListener: InferenceTimeListener?
    private var currentOnFpsRateListener: FpsRateListener?
    private var screenSize: CGSize?
    var t0 = 0.0  // inference start
    var t1 = 0.0  // inference dt
    var t2 = 0.0  // inference dt smoothed
    var t3 = CACurrentMediaTime()  // FPS start
    var t4 = 0.0  // FPS dt smoothed

    public init?(poseModel: any YoloModel) async throws {
        if poseModel.task != "pose" {
            throw PredictorError.invalidTask
        }

        guard let mlModel = try await poseModel.loadModel() as? MLModel
        else { return }

        let bounds: CGRect = await UIScreen.main.bounds
        screenSize = CGSize(width: bounds.width, height: bounds.height)

        poseModel = try! VNCoreMLModel(for: mlModel)

        visionRequest = {
            let request = VNCoreMLRequest(
                model: poseModel,
                completionHandler: { [weak self] request, error in
                    self?.processObservations(for: request, error: error)
                }
            )
            request.imageCropAndScaleOption = .scaleFill
            return request
        }()
    }

    public func predict(
        sampleBuffer: CMSampleBuffer,
        onResultsListener: ResultsListener?,
        onInferenceTime: InferenceTimeListener?,
        onFpsRate: FpsRateListener?
    ) {
        guard currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        currentBuffer = pixelBuffer
        currentOnResultsListener = onResultsListener
        currentOnInferenceTimeListener = onInferenceTime
        currentOnFpsRateListener = onFpsRate

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        t0 = CACurrentMediaTime()  // inference start
        do {
            try handler.perform([visionRequest!])
        } catch {
            print("Pose estimation error: \(error)")
        }
        t1 = CACurrentMediaTime() - t0  // inference dt

        currentBuffer = nil
    }

    private func processObservations(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            guard let results = request.results as? [VNRecognizedPointsObservation] else {
                print("Pose estimation failed")
                return
            }

            var keypointsData: [[String: Any]] = []

            for observation in results {
                if let points = try? observation.recognizedPoints(forGroupKey: .all) {
                    let keypoints = points.mapValues { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
                    keypointsData.append(["keypoints": keypoints])
                }
            }

            // Send results to the listener
            self.currentOnResultsListener?.on(predictions: keypointsData)

            // Measure FPS
            if self.t1 < 10.0 {  // valid dt
                self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
            }
            self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed FPS
            self.t3 = CACurrentMediaTime()

            self.currentOnInferenceTimeListener?.on(inferenceTime: self.t2 * 1000)  // ms
            self.currentOnFpsRateListener?.on(fpsRate: 1 / self.t4)
        }
    }
}
