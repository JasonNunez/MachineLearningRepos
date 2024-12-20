using System.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEditor;
using Unity.Sentis;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Inference;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;

namespace Unity.MLAgents.Tests
{
    public class Test3DSensorComponent : SensorComponent
    {
        public ISensor Sensor;

        public override ISensor[] CreateSensors()
        {
            return new ISensor[] { Sensor };
        }
    }

    public class Test3DSensor : ISensor, IBuiltInSensor
    {
        int m_Width;
        int m_Height;
        int m_Channels;
        string m_Name;
        // Dummy value for the IBuiltInSensor interface
        public const int k_BuiltInSensorType = -42;

        public Test3DSensor(string name, int width, int height, int channels)
        {
            m_Width = width;
            m_Height = height;
            m_Channels = channels;
            m_Name = name;
        }

        public ObservationSpec GetObservationSpec()
        {
            return ObservationSpec.Visual(m_Channels, m_Height, m_Width);
        }

        public int Write(ObservationWriter writer)
        {
            for (int i = 0; i < m_Width * m_Height * m_Channels; i++)
            {
                writer[i] = 0.0f;
            }
            return m_Width * m_Height * m_Channels;
        }

        public byte[] GetCompressedObservation()
        {
            return new byte[0];
        }

        public void Update() { }
        public void Reset() { }

        public CompressionSpec GetCompressionSpec()
        {
            return CompressionSpec.Default();
        }

        public string GetName()
        {
            return m_Name;
        }

        public BuiltInSensorType GetBuiltInSensorType()
        {
            return (BuiltInSensorType)k_BuiltInSensorType;
        }
    }

    [TestFixture]
    public class ParameterLoaderTest
    {
        const string k_discrete_ONNX_v2 = "Packages/com.unity.ml-agents/Tests/Editor/TestModels/discrete_rank2_vector_v2_0.onnx";
        const string k_hybrid_ONNX_recurr_v2 = "Packages/com.unity.ml-agents/Tests/Editor/TestModels/hybrid0vis8vec_2c_2_3d_v2_0.onnx";


        // ONNX model with continuous/discrete action output (support hybrid action)
        const string k_continuousONNXPath = "Packages/com.unity.ml-agents/Tests/Editor/TestModels/continuous2vis8vec2action_v1_0.onnx";
        const string k_discreteONNXPath = "Packages/com.unity.ml-agents/Tests/Editor/TestModels/discrete1vis0vec_2_3action_obsolete_recurr_v1_0.onnx";
        const string k_hybridONNXPath = "Packages/com.unity.ml-agents/Tests/Editor/TestModels/hybrid0vis53vec_3c_2daction_v1_0.onnx";
        // NN model with single action output (deprecated, does not support hybrid action).
        // Same BrainParameters settings as the corresponding ONNX model.

        ModelAsset rank2ONNXModel;
        ModelAsset hybridRecurrV2Model;
        ModelAsset continuousONNXModel;
        ModelAsset discreteONNXModel;
        ModelAsset hybridONNXModel;
        Test3DSensorComponent sensor_21_20_3;
        Test3DSensorComponent sensor_20_22_3;
        BufferSensor sensor_23_20;
        VectorSensor sensor_8;
        VectorSensor sensor_10;

        BrainParameters GetContinuous2vis8vec2actionBrainParameters()
        {
            var validBrainParameters = new BrainParameters();
            validBrainParameters.VectorObservationSize = 8;
            validBrainParameters.NumStackedVectorObservations = 1;
            validBrainParameters.ActionSpec = ActionSpec.MakeContinuous(2);
            return validBrainParameters;
        }

        BrainParameters GetDiscrete1vis0vec_2_3action_recurrModelBrainParameters()
        {
            var validBrainParameters = new BrainParameters();
            validBrainParameters.VectorObservationSize = 0;
            validBrainParameters.NumStackedVectorObservations = 1;
            validBrainParameters.ActionSpec = ActionSpec.MakeDiscrete(2, 3);
            return validBrainParameters;
        }

        BrainParameters GetHybridBrainParameters()
        {
            var validBrainParameters = new BrainParameters();
            validBrainParameters.VectorObservationSize = 53;
            validBrainParameters.NumStackedVectorObservations = 1;
            validBrainParameters.ActionSpec = new ActionSpec(3, new[] { 2 });
            return validBrainParameters;
        }

        BrainParameters GetRank2BrainParameters()
        {
            var validBrainParameters = new BrainParameters();
            validBrainParameters.VectorObservationSize = 4;
            validBrainParameters.NumStackedVectorObservations = 2;
            validBrainParameters.ActionSpec = ActionSpec.MakeDiscrete(3, 3, 3);
            return validBrainParameters;
        }

        BrainParameters GetRecurrHybridBrainParameters()
        {
            var validBrainParameters = new BrainParameters();
            validBrainParameters.VectorObservationSize = 8;
            validBrainParameters.NumStackedVectorObservations = 1;
            validBrainParameters.ActionSpec = new ActionSpec(2, new int[] { 2, 3 });
            return validBrainParameters;
        }

        [SetUp]
        public void SetUp()
        {
            continuousONNXModel = (ModelAsset)AssetDatabase.LoadAssetAtPath(k_continuousONNXPath, typeof(ModelAsset));
            discreteONNXModel = (ModelAsset)AssetDatabase.LoadAssetAtPath(k_discreteONNXPath, typeof(ModelAsset));
            hybridONNXModel = (ModelAsset)AssetDatabase.LoadAssetAtPath(k_hybridONNXPath, typeof(ModelAsset));
            rank2ONNXModel = (ModelAsset)AssetDatabase.LoadAssetAtPath(k_discrete_ONNX_v2, typeof(ModelAsset));
            hybridRecurrV2Model = (ModelAsset)AssetDatabase.LoadAssetAtPath(k_hybrid_ONNX_recurr_v2, typeof(ModelAsset));
            var go = new GameObject("SensorA");
            sensor_21_20_3 = go.AddComponent<Test3DSensorComponent>();
            sensor_21_20_3.Sensor = new Test3DSensor("SensorA", 21, 20, 3);
            sensor_20_22_3 = go.AddComponent<Test3DSensorComponent>();
            sensor_20_22_3.Sensor = new Test3DSensor("SensorA", 20, 22, 3);
            sensor_23_20 = new BufferSensor(20, 23, "BufferSensor");
            sensor_8 = new VectorSensor(8, "VectorSensor8");
            sensor_10 = new VectorSensor(10, "VectorSensor10");
        }

        [Test]
        public void TestModelExist()
        {
            Assert.IsNotNull(continuousONNXModel);
            Assert.IsNotNull(discreteONNXModel);
            Assert.IsNotNull(hybridONNXModel);
            Assert.IsNotNull(rank2ONNXModel);
            Assert.IsNotNull(hybridRecurrV2Model);
        }

        [Test]
        public void TestGetInputTensorsContinuous()
        {
            var model = ModelLoader.Load(continuousONNXModel);
            var modelInfo = new SentisModelInfo(model);
            var inputNames = modelInfo.InputNames;
            // Model should contain 3 inputs : vector, visual 1 and visual 2
            Assert.AreEqual(3, inputNames.Count());
            Assert.Contains(TensorNames.VectorObservationPlaceholder, inputNames);
            Assert.Contains(TensorNames.VisualObservationPlaceholderPrefix + "0", inputNames);
            Assert.Contains(TensorNames.VisualObservationPlaceholderPrefix + "1", inputNames);

            Assert.AreEqual(2, modelInfo.NumVisualInputs);

            modelInfo.Dispose();
        }

        public void TestGetInputTensorsDiscrete()
        {
            var model = ModelLoader.Load(discreteONNXModel);
            var modelInfo = new SentisModelInfo(model);
            var inputNames = modelInfo.InputNames;
            // Model should contain 2 inputs : recurrent and visual 1

            Assert.Contains(TensorNames.VisualObservationPlaceholderPrefix + "0", inputNames);
            // TODO :There are some memory tensors as well
            modelInfo.Dispose();
        }

        [Test]
        public void TestGetInputTensorsHybrid()
        {
            var model = ModelLoader.Load(hybridONNXModel);
            var modelInfo = new SentisModelInfo(model);
            var inputNames = modelInfo.InputNames;
            Assert.Contains(TensorNames.VectorObservationPlaceholder, inputNames);
            modelInfo.Dispose();
        }

        [Test]
        public void TestGetOutputTensorsContinuous()
        {
            var model = ModelLoader.Load(continuousONNXModel);
            var modelInfo = new SentisModelInfo(model);
            var outputNames = modelInfo.OutputNames;
            var actionOutputName = TensorNames.ContinuousActionOutput;
            Assert.Contains(actionOutputName, outputNames);
            Assert.AreEqual(1, outputNames.Count());
            modelInfo.Dispose();
        }

        [Test]
        public void TestGetOutputTensorsDiscrete()
        {
            var model = ModelLoader.Load(discreteONNXModel);
            var modelInfo = new SentisModelInfo(model);
            var outputNames = modelInfo.OutputNames;
            var actionOutputName = TensorNames.DiscreteActionOutput;
            Assert.Contains(actionOutputName, outputNames);
            // TODO : There are some memory tensors as well
            modelInfo.Dispose();
        }

        [Test]
        public void TestGetOutputTensorsHybrid()
        {
            var model = ModelLoader.Load(hybridONNXModel);
            var modelInfo = new SentisModelInfo(model);
            var outputNames = modelInfo.OutputNames;

            Assert.AreEqual(2, outputNames.Count());
            Assert.Contains(TensorNames.ContinuousActionOutput, outputNames);
            Assert.Contains(TensorNames.DiscreteActionOutput, outputNames);

            modelInfo.Dispose();
        }

        [Test]
        public void TestCheckModelRank2()
        {
            var model = ModelLoader.Load(rank2ONNXModel);
            var validBrainParameters = GetRank2BrainParameters();

            var errors = SentisModelParamLoader.CheckModel(
                model, validBrainParameters,
                new ISensor[] { sensor_23_20, sensor_10, sensor_8 }, new ActuatorComponent[0]
            );
            Assert.AreEqual(0, errors.Count()); // There should not be any errors

            errors = SentisModelParamLoader.CheckModel(
                model, validBrainParameters,
                new ISensor[] { sensor_23_20, sensor_10 }, new ActuatorComponent[0]
            );
            Assert.AreNotEqual(0, errors.Count()); // Wrong number of sensors

            errors = SentisModelParamLoader.CheckModel(
                model, validBrainParameters,
                new ISensor[] { new BufferSensor(20, 40, "BufferSensor"), sensor_10, sensor_8 }, new ActuatorComponent[0]
            );
            Assert.AreNotEqual(0, errors.Count()); // Wrong buffer sensor size

            errors = SentisModelParamLoader.CheckModel(
                model, validBrainParameters,
                new ISensor[] { sensor_23_20, sensor_10, sensor_10 }, new ActuatorComponent[0]
            );
            Assert.AreNotEqual(0, errors.Count()); // Wrong vector sensor size
        }

        [Test]
        public void TestCheckModelValidContinuous()
        {
            var model = ModelLoader.Load(continuousONNXModel);
            var validBrainParameters = GetContinuous2vis8vec2actionBrainParameters();

            var errors = SentisModelParamLoader.CheckModel(
                model, validBrainParameters,
                new ISensor[]
                {
                    new VectorSensor(8),
                    sensor_21_20_3.CreateSensors()[0],
                    sensor_20_22_3.CreateSensors()[0]
                },
                new ActuatorComponent[0]
            );
            Assert.AreEqual(0, errors.Count()); // There should not be any errors
        }

        [Test]
        public void TestCheckModelValidDiscrete()
        {
            var model = ModelLoader.Load(discreteONNXModel);
            var validBrainParameters = GetDiscrete1vis0vec_2_3action_recurrModelBrainParameters();

            var errors = SentisModelParamLoader.CheckModel(
                model, validBrainParameters,
                new ISensor[] { sensor_21_20_3.CreateSensors()[0] }, new ActuatorComponent[0]
            );
            foreach (var e in errors)
            {
                Debug.Log(e.Message);
            }
            Assert.Greater(errors.Count(), 0); // There should be an error since LSTM v1.x is not supported
        }

        [Test]
        public void TestCheckModelValidRecurrent()
        {
            var model = ModelLoader.Load(hybridRecurrV2Model);
            var num_errors = 0; // A model trained with v2 should not raise errors
            var validBrainParameters = GetRecurrHybridBrainParameters();

            var errors = SentisModelParamLoader.CheckModel(
                model, validBrainParameters,
                new ISensor[] { sensor_8 }, new ActuatorComponent[0]
            );
            Assert.AreEqual(num_errors, errors.Count()); // There should not be any errors

            var invalidBrainParameters = GetRecurrHybridBrainParameters();
            invalidBrainParameters.ActionSpec = new ActionSpec(1, new int[] { 2, 3 });
            errors = SentisModelParamLoader.CheckModel(
                model, invalidBrainParameters,
                new ISensor[] { sensor_8 }, new ActuatorComponent[0]
            );
            Assert.AreEqual(1, errors.Count()); // 1 continuous action instead of 2

            invalidBrainParameters.ActionSpec = new ActionSpec(2, new int[] { 3, 2 });
            errors = SentisModelParamLoader.CheckModel(
                model, invalidBrainParameters,
                new ISensor[] { sensor_8 }, new ActuatorComponent[0]
            );
            Assert.AreEqual(1, errors.Count()); // Discrete action branches flipped
        }

        [Test]
        public void TestCheckModelValidHybrid()
        {
            var model = ModelLoader.Load(hybridONNXModel);
            var validBrainParameters = GetHybridBrainParameters();

            var errors = SentisModelParamLoader.CheckModel(
                model, validBrainParameters,
                new ISensor[]
                {
                    new VectorSensor(validBrainParameters.VectorObservationSize)
                }, new ActuatorComponent[0]
            );
            Assert.AreEqual(0, errors.Count()); // There should not be any errors
        }

        [Test]
        public void TestCheckModelThrowsVectorObservationContinuous()
        {
            var model = ModelLoader.Load(continuousONNXModel);

            var brainParameters = GetContinuous2vis8vec2actionBrainParameters();
            brainParameters.VectorObservationSize = 9; // Invalid observation
            var errors = SentisModelParamLoader.CheckModel(
                model, brainParameters,
                new ISensor[]
                {
                    sensor_21_20_3.CreateSensors()[0],
                    sensor_20_22_3.CreateSensors()[0]
                },
                new ActuatorComponent[0]
            );
            Assert.Greater(errors.Count(), 0);

            brainParameters = GetContinuous2vis8vec2actionBrainParameters();
            brainParameters.NumStackedVectorObservations = 2;// Invalid stacking
            errors = SentisModelParamLoader.CheckModel(
                model, brainParameters,
                new ISensor[]
                {
                    sensor_21_20_3.CreateSensors()[0],
                    sensor_20_22_3.CreateSensors()[0]
                },
                new ActuatorComponent[0]
            );
            Assert.Greater(errors.Count(), 0);
        }

        [Test]
        public void TestCheckModelThrowsVectorObservationDiscrete()
        {
            var model = ModelLoader.Load(discreteONNXModel);

            var brainParameters = GetDiscrete1vis0vec_2_3action_recurrModelBrainParameters();
            brainParameters.VectorObservationSize = 1; // Invalid observation
            var errors = SentisModelParamLoader.CheckModel(
                model, brainParameters, new ISensor[]
                {
                    sensor_21_20_3.CreateSensors()[0]
                },
                new ActuatorComponent[0]
            );
            Assert.Greater(errors.Count(), 0);
        }

        [Test]
        public void TestCheckModelThrowsVectorObservationHybrid()
        {
            var model = ModelLoader.Load(hybridONNXModel);

            var brainParameters = GetHybridBrainParameters();
            brainParameters.VectorObservationSize = 9; // Invalid observation
            var errors = SentisModelParamLoader.CheckModel(
                model, brainParameters,
                new ISensor[] { }, new ActuatorComponent[0]
            );
            Assert.Greater(errors.Count(), 0);

            brainParameters = GetContinuous2vis8vec2actionBrainParameters();
            brainParameters.NumStackedVectorObservations = 2;// Invalid stacking
            errors = SentisModelParamLoader.CheckModel(
                model, brainParameters,
                new ISensor[] { }, new ActuatorComponent[0]
            );
            Assert.Greater(errors.Count(), 0);
        }

        [Test]
        public void TestCheckModelThrowsActionContinuous()
        {
            var model = ModelLoader.Load(continuousONNXModel);

            var brainParameters = GetContinuous2vis8vec2actionBrainParameters();
            brainParameters.ActionSpec = ActionSpec.MakeContinuous(3); // Invalid action
            var errors = SentisModelParamLoader.CheckModel(
                model, brainParameters, new ISensor[]
                {
                    sensor_21_20_3.CreateSensors()[0],
                    sensor_20_22_3.CreateSensors()[0]
                },
                new ActuatorComponent[0]
            );
            Assert.Greater(errors.Count(), 0);

            brainParameters = GetContinuous2vis8vec2actionBrainParameters();
            brainParameters.ActionSpec = ActionSpec.MakeDiscrete(3); // Invalid SpaceType
            errors = SentisModelParamLoader.CheckModel(
                model, brainParameters, new ISensor[]
                {
                    sensor_21_20_3.CreateSensors()[0],
                    sensor_20_22_3.CreateSensors()[0]
                },
                new ActuatorComponent[0]
            );
            Assert.Greater(errors.Count(), 0);
        }

        [Test]
        public void TestCheckModelThrowsActionDiscrete()
        {
            var model = ModelLoader.Load(discreteONNXModel);

            var brainParameters = GetDiscrete1vis0vec_2_3action_recurrModelBrainParameters();
            brainParameters.ActionSpec = ActionSpec.MakeDiscrete(3, 3); // Invalid action
            var errors = SentisModelParamLoader.CheckModel(
                model, brainParameters,
                new ISensor[] { sensor_21_20_3.CreateSensors()[0] },
                new ActuatorComponent[0]
            );
            Assert.Greater(errors.Count(), 0);

            brainParameters = GetContinuous2vis8vec2actionBrainParameters();
            brainParameters.ActionSpec = ActionSpec.MakeContinuous(2); // Invalid SpaceType
            errors = SentisModelParamLoader.CheckModel(
                model,
                brainParameters,
                new ISensor[] { sensor_21_20_3.CreateSensors()[0] },
                new ActuatorComponent[0]
            );
            Assert.Greater(errors.Count(), 0);
        }

        [Test]
        public void TestCheckModelThrowsActionHybrid()
        {
            var model = ModelLoader.Load(hybridONNXModel);

            var brainParameters = GetHybridBrainParameters();
            brainParameters.ActionSpec = new ActionSpec(3, new[] { 3 }); // Invalid discrete action size
            var errors = SentisModelParamLoader.CheckModel(
                model,
                brainParameters,
                new ISensor[]
                {
                    sensor_21_20_3.CreateSensors()[0],
                    sensor_20_22_3.CreateSensors()[0]
                },
                new ActuatorComponent[0]
            );
            Assert.Greater(errors.Count(), 0);

            brainParameters = GetContinuous2vis8vec2actionBrainParameters();
            brainParameters.ActionSpec = ActionSpec.MakeDiscrete(2); // Missing continuous action
            errors = SentisModelParamLoader.CheckModel(
                model,
                brainParameters,
                new ISensor[]
                {
                    sensor_21_20_3.CreateSensors()[0],
                    sensor_20_22_3.CreateSensors()[0]
                },
                new ActuatorComponent[0]
            );
            Assert.Greater(errors.Count(), 0);
        }

        [Test]
        public void TestCheckModelThrowsNoModel()
        {
            var brainParameters = GetContinuous2vis8vec2actionBrainParameters();
            var errors = SentisModelParamLoader.CheckModel(
                null,
                brainParameters,
                new ISensor[]
                {
                    sensor_21_20_3.CreateSensors()[0],
                    sensor_20_22_3.CreateSensors()[0]
                },
                new ActuatorComponent[0]
            );
            Assert.Greater(errors.Count(), 0);
        }
    }
}
