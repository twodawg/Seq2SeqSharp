using AdvUtils;
using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Utils;

var training = ModeEnums.Test; // Toggle depending if a retraining is needed (did the statements change?) - is the model name correct?
var rootDir = "D:\\Temp\\CC250\\";
// Define model parameters
// CPU training 26m per epoch
// 18 s per epoch on GPU (release) 32 parallel 16 batch 12 encoder depth
// 25 s per epoch on GPU (release) 16 parallel 16 batch 24 encoder depth - 40m
var opts = new SeqClassificationOptions
{
    ModelFilePath = training == ModeEnums.Train ? $"{rootDir}model.bin" : $"{rootDir}model.bin.5100",
    SrcLang = "Src",
    TgtLang = "Labels",
    ProcessorType = ProcessorTypeEnums.GPU,
    AttentionType = AttentionTypeEnums.FlashAttentionV2,
    TrainCorpusPath = $"{rootDir}train",
    ValidCorpusPaths = $"{rootDir}valid",
    LogDestination = Logger.Destination.Console,
    TaskParallelism = 16, //32,             

    MaxEpochNum = 100,
    MaxSentLength = 1024,
    BatchSize = 16,
    //StartLearningRate = 0.01f,
    EncoderLayerDepth = 24, // 12 Increases the model size

    Task = training,
    InputTestFile = $"{rootDir}input.txt",
    OutputFile = $"{rootDir}output.txt",
    LogLevel = Logger.Level.info,
};
Logger.Initialize(opts.LogDestination, opts.LogLevel, $"{opts.Task}_{Utils.GetTimeStamp(DateTime.Now)}.log");
do
{
    string prompt = null;
    if (training == ModeEnums.Test)
    {
        Console.WriteLine("What are your symptoms?");
        prompt = Console.ReadLine();
    }

    DecodingOptions decodingOptions = opts.CreateDecodingOptions();
    SeqClassification ss = null;

    if (opts.Task == ModeEnums.Train)
    {
        List<string> inputSentences = [.. File.ReadAllLines($"{rootDir}\\train\\PatientCareReportNarratives.txt")];
        // Cannot contain spaces
        List<string> labels = [.. File.ReadAllLines($"{rootDir}\\train\\ConditionCodes.txt")];
        // Save each labels with a tab and then each inputSentences to D:\Temp\train.enu.snt
        Directory.CreateDirectory(opts.TrainCorpusPath);
        File.WriteAllLines($"{opts.TrainCorpusPath}\\train.src.snt", inputSentences.Select((sentence, index) => $"{sentence}"));
        File.WriteAllLines($"{opts.TrainCorpusPath}\\train.labels.snt", inputSentences.Select((sentence, index) => $"{labels[index]}"));
                
        // Prepare data for Seq2SeqSharp
        // Load train corpus
        var trainCorpus = new SeqClassificationMultiTasksCorpus(corpusFilePath: opts.TrainCorpusPath,
          srcLangName: opts.SrcLang, tgtLangName: opts.TgtLang, maxTokenSizePerBatch: opts.MaxTokenSizePerBatch,
          maxSentLength: opts.MaxSentLength, paddingEnums: opts.PaddingType, tooLongSequence: opts.TooLongSequence);

        // Valid corpus
        var validCorpusList = new List<SeqClassificationMultiTasksCorpus>();
        validCorpusList.Add(new SeqClassificationMultiTasksCorpus(opts.ValidCorpusPaths,
          srcLangName: opts.SrcLang, tgtLangName: opts.TgtLang, opts.ValMaxTokenSizePerBatch,
          opts.MaxSentLength, paddingEnums: opts.PaddingType, tooLongSequence: opts.TooLongSequence));

        // Create optimizer
        IOptimizer optimizer = Misc.CreateOptimizer(opts);

        var (srcVocab, tgtVocab) = trainCorpus.BuildVocabs(opts.SrcVocabSize, opts.TgtVocabSize);

        // Create metrics
        Dictionary<int, List<IMetric>> taskId2metrics = new();
        taskId2metrics.Add(0, new List<IMetric>());
        taskId2metrics[0].Add(new MultiLabelsFscoreMetric("", tgtVocab.GetAllTokens(keepBuildInTokens: false)));

        // Create learning rate
        ILearningRate learningRate = new DecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, opts.WeightsUpdateCount,
          opts.LearningRateStepDownFactor, opts.UpdateNumToStepDownLearningRate);

        // Train the model
        ss = new SeqClassification(opts, srcVocab, tgtVocab);

        // Add event handler for monitoring
        ss.StatusUpdateWatcher += Misc.Ss_StatusUpdateWatcher;
        ss.EvaluationWatcher += Ss_EvaluationWatcher;

        ss.Train(opts.MaxEpochNum, trainCorpus, validCorpusList.ToArray(),
        learningRate, taskId2metrics: taskId2metrics, optimizer, decodingOptions);

        Console.WriteLine("Training complete!");
    }
    else if (opts.Task == ModeEnums.Valid)
    {
        List<string> inputSentences = [.. File.ReadAllLines($"{rootDir}\\valid\\PatientCareReportNarratives.txt")];
        // Cannot contain spaces
        List<string> labels = [.. File.ReadAllLines($"{rootDir}\\valid\\ConditionCodes.txt")];
        Directory.CreateDirectory(opts.ValidCorpusPaths);
        File.WriteAllLines($"{opts.ValidCorpusPaths}\\valid.src.snt", inputSentences.Select((sentence, index) => $"{sentence}"));
        File.WriteAllLines($"{opts.ValidCorpusPaths}\\valid.labels.snt", inputSentences.Select((sentence, index) => $"{labels[index]}"));

        Logger.WriteLine($"Evaluate model '{opts.ModelFilePath}' by valid corpus '{opts.ValidCorpusPaths}'");

        // Create metrics
        ss = new SeqClassification(opts);
        Dictionary<int, List<IMetric>> taskId2metrics = new Dictionary<int, List<IMetric>>();
        taskId2metrics.Add(0, new List<IMetric>());
        taskId2metrics[0].Add(new MultiLabelsFscoreMetric("", ss.TgtVocab.GetAllTokens(keepBuildInTokens: false)));

        ss = new SeqClassification(opts);
        ss.EvaluationWatcher += Ss_EvaluationWatcher;

        // Load valid corpus
        if (!opts.ValidCorpusPaths.IsNullOrEmpty())
        {
            Logger.WriteLine($"Loading valid corpus '{opts.ValidCorpusPaths}'");
            var validCorpus = new SeqClassificationMultiTasksCorpus(opts.ValidCorpusPaths, srcLangName: opts.SrcLang, tgtLangName: opts.TgtLang, opts.ValMaxTokenSizePerBatch, opts.MaxSentLength, paddingEnums: opts.PaddingType, tooLongSequence: opts.TooLongSequence);

            Logger.WriteLine($"Validating corpus '{opts.ValidCorpusPaths}'");
            ss.Valid(validCorpus, taskId2metrics, null);
        }
        break;
    }
    else if (opts.Task == ModeEnums.Test)
    {
        opts.ProcessorType = ProcessorTypeEnums.CPU;
        opts.AttentionType = AttentionTypeEnums.Classic;

        File.WriteAllText(opts.InputTestFile, prompt);

        if (File.Exists(opts.OutputFile))
        {
            File.Delete(opts.OutputFile);
        }

        //Test trained model
        ss = new SeqClassification(opts);
        //Stopwatch stopwatch = Stopwatch.StartNew();

        ss.Test<SeqClassificationMultiTasksCorpusBatch>(opts.InputTestFile, opts.OutputFile, opts.BatchSize, decodingOptions, opts.SrcSentencePieceModelPath, opts.TgtSentencePieceModelPath);

        //stopwatch.Stop();

        //Logger.WriteLine($"Test mode execution time elapsed: '{stopwatch.Elapsed}'");

        Console.WriteLine("***\nAns: " + File.ReadAllText(opts.OutputFile));
    }
    void Ss_EvaluationWatcher(object sender, EventArgs e)
    {
        EvaluationEventArg ep = e as EvaluationEventArg;

        Logger.WriteLine(Logger.Level.info, ep.Color, ep.Message);
    }
}
while (training == ModeEnums.Test);

