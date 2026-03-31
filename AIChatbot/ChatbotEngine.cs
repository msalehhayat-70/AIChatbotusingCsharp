using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Win32;

namespace AIChatbot
{
    /// <summary>
    /// Core AI engine — bridges C# WinForms to the trained scikit-learn
    /// TF-IDF + LinearSVC model via a Python subprocess call.
    /// </summary>
    public class ChatbotEngine
    {
        private readonly string _pythonExe;
        private readonly string _scriptPath;
        private bool _isReady = false;

        public bool IsReady => _isReady;
        public string PythonPath => _pythonExe;

        public ChatbotEngine()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            _scriptPath = Path.Combine(baseDir, "Python", "predict.py");
            _pythonExe  = FindPython();
        }

        // ─────────────────────────────────────────────────────
        //  INITIALISE — verify Python + packages + model
        // ─────────────────────────────────────────────────────
        public async Task<(bool success, string message)> InitialiseAsync()
        {
            try
            {
                // 1. Python found?
                if (string.IsNullOrEmpty(_pythonExe))
                    return (false,
                        "❌ Python not found on this machine.\n\n" +
                        "Please:\n" +
                        "1. Download Python from https://python.org\n" +
                        "2. During install CHECK 'Add Python to PATH'\n" +
                        "3. Restart Visual Studio after installing\n\n" +
                        "Then run:  pip install scikit-learn scipy numpy");

                // 2. predict.py exists?
                if (!File.Exists(_scriptPath))
                    return (false,
                        $"❌ Inference script missing.\n\nExpected at:\n{_scriptPath}\n\n" +
                        "Make sure the Python\\ folder is in the project output directory.");

                // 3. Model file exists?
                string modelPath = Path.Combine(
                    AppDomain.CurrentDomain.BaseDirectory, "Model", "chatbot_model.pkl");
                if (!File.Exists(modelPath))
                    return (false,
                        $"❌ Trained model file missing.\n\nExpected at:\n{modelPath}\n\n" +
                        "Make sure the Model\\ folder is in the project output directory.");

                // 4. Verify scikit-learn + scipy importable
                string verifyResult = await RunPythonAsync(
                    "-c \"import sklearn, scipy, pickle; print('PACKAGES_OK')\"");

                if (!verifyResult.Contains("PACKAGES_OK"))
                {
                    // Try to give a helpful pip command with the exact python path
                    string pipCmd = _pythonExe.EndsWith(".exe", StringComparison.OrdinalIgnoreCase)
                        ? $"\"{_pythonExe}\" -m pip install scikit-learn scipy numpy"
                        : $"{_pythonExe} -m pip install scikit-learn scipy numpy";

                    return (false,
                        "❌ Required Python packages are missing.\n\n" +
                        $"Python found at:\n  {_pythonExe}\n\n" +
                        "Open Command Prompt and run:\n" +
                        $"  {pipCmd}\n\n" +
                        "Then restart the application.");
                }

                // 5. Warm-up prediction
                var warmup = await PredictAsync("hello");
                if (warmup.Intent == "error")
                    return (false, $"❌ Model warm-up failed:\n{warmup.Response}");

                _isReady = true;
                return (true, $"✅ TechBot AI engine loaded!\nPython: {_pythonExe}");
            }
            catch (Exception ex)
            {
                return (false, $"❌ Initialisation error:\n{ex.Message}");
            }
        }

        // ─────────────────────────────────────────────────────
        //  PREDICT — classify user input via trained model
        // ─────────────────────────────────────────────────────
        public async Task<ChatbotResponse> PredictAsync(string userInput)
        {
            if (string.IsNullOrWhiteSpace(userInput))
                return new ChatbotResponse
                {
                    Intent   = "unknown",
                    Response = "Please type something so I can help you!",
                    Confidence = 0
                };

            try
            {
                // Sanitise for command-line passing
                string escaped = userInput
                    .Replace("\\", "\\\\")
                    .Replace("\"", "\\\"")
                    .Replace("\n", " ")
                    .Replace("\r", "");

                string args   = $"\"{_scriptPath}\" \"{escaped}\"";
                string output = await RunPythonAsync(args);

                if (string.IsNullOrWhiteSpace(output))
                    return FallbackResponse();

                // Find JSON in output (ignore any warning lines before it)
                string json = ExtractJson(output);
                if (string.IsNullOrEmpty(json))
                    return FallbackResponse();

                var doc  = JsonDocument.Parse(json);
                var root = doc.RootElement;

                return new ChatbotResponse
                {
                    Intent     = root.GetProperty("intent").GetString()     ?? "unknown",
                    Response   = root.GetProperty("response").GetString()   ?? "Sorry, no response generated.",
                    Confidence = root.GetProperty("confidence").GetDouble()
                };
            }
            catch (Exception ex)
            {
                return new ChatbotResponse
                {
                    Intent   = "error",
                    Response = $"Prediction error: {ex.Message}",
                    Confidence = 0
                };
            }
        }

        // ─────────────────────────────────────────────────────
        //  HELPERS
        // ─────────────────────────────────────────────────────

        private async Task<string> RunPythonAsync(string args)
        {
            var psi = new ProcessStartInfo
            {
                FileName               = _pythonExe,
                Arguments              = args,
                RedirectStandardOutput = true,
                RedirectStandardError  = true,
                UseShellExecute        = false,
                CreateNoWindow         = true,
                StandardOutputEncoding = Encoding.UTF8
            };

            using var process = new Process { StartInfo = psi };
            var stdout = new StringBuilder();
            var stderr = new StringBuilder();

            process.OutputDataReceived += (_, e) => { if (e.Data != null) stdout.AppendLine(e.Data); };
            process.ErrorDataReceived  += (_, e) => { if (e.Data != null) stderr.AppendLine(e.Data); };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            bool finished = await Task.Run(() => process.WaitForExit(20000));
            if (!finished) { try { process.Kill(); } catch { } return string.Empty; }

            return stdout.ToString().Trim();
        }

        /// <summary>
        /// Extract the first {...} JSON object from output that may contain
        /// Python deprecation warnings or other non-JSON lines before it.
        /// </summary>
        private static string ExtractJson(string output)
        {
            int start = output.IndexOf('{');
            int end   = output.LastIndexOf('}');
            if (start >= 0 && end > start)
                return output[start..(end + 1)];
            return string.Empty;
        }

        /// <summary>
        /// Smart Python finder — checks PATH, Windows registry, and common
        /// install locations so it works regardless of how Python was installed.
        /// </summary>
        private static string FindPython()
        {
            // ── 0. Hardcoded known path for this machine ────────────────
            // Python 3.14 confirmed installed at this location
            string knownPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                @"Programs\Python\Python314\python.exe");
            if (File.Exists(knownPath) && TryPython(knownPath))
                return knownPath;
            string[] pathNames = { "python", "python3", "python3.13", "python3.12",
                                   "python3.11", "python3.10", "python3.9", "python3.8" };
            foreach (var name in pathNames)
            {
                if (TryPython(name)) return name;
            }

            // ── 2. Check Windows Registry (py launcher & direct installs) ─
            if (OperatingSystem.IsWindows())
            {
                string? regPath = FindPythonInRegistry();
                if (regPath != null && TryPython(regPath)) return regPath;
            }

            // ── 3. Common Windows install locations ─────────────────────
            string userProfile  = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            string programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
            string localApp     = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);

            string[] commonPaths =
            {
                // Windows Store Python
                Path.Combine(localApp, @"Microsoft\WindowsApps\python.exe"),
                Path.Combine(localApp, @"Microsoft\WindowsApps\python3.exe"),
                // ── Python 3.14 (current user install — confirmed on this machine) ──
                Path.Combine(localApp, @"Programs\Python\Python314\python.exe"),
                Path.Combine(programFiles, @"Python314\python.exe"),
                // ── Python 3.13 ──
                Path.Combine(localApp, @"Programs\Python\Python313\python.exe"),
                Path.Combine(programFiles, @"Python313\python.exe"),
                // ── Python 3.12 ──
                Path.Combine(localApp, @"Programs\Python\Python312\python.exe"),
                Path.Combine(programFiles, @"Python312\python.exe"),
                // ── Python 3.11 ──
                Path.Combine(localApp, @"Programs\Python\Python311\python.exe"),
                Path.Combine(programFiles, @"Python311\python.exe"),
                // ── Python 3.10 ──
                Path.Combine(localApp, @"Programs\Python\Python310\python.exe"),
                Path.Combine(programFiles, @"Python310\python.exe"),
                // ── Python 3.9 / 3.8 ──
                Path.Combine(localApp, @"Programs\Python\Python39\python.exe"),
                Path.Combine(localApp, @"Programs\Python\Python38\python.exe"),
                Path.Combine(programFiles, @"Python39\python.exe"),
                Path.Combine(programFiles, @"Python38\python.exe"),
                // Anaconda / Miniconda
                Path.Combine(userProfile, @"anaconda3\python.exe"),
                Path.Combine(userProfile, @"miniconda3\python.exe"),
                Path.Combine(userProfile, @"Anaconda3\python.exe"),
                Path.Combine(userProfile, @"Miniconda3\python.exe"),
                @"C:\anaconda3\python.exe",
                @"C:\miniconda3\python.exe",
                @"C:\ProgramData\anaconda3\python.exe",
                @"C:\ProgramData\miniconda3\python.exe",
            };

            foreach (var fullPath in commonPaths)
            {
                if (File.Exists(fullPath) && TryPython(fullPath))
                    return fullPath;
            }

            // ── 4. Last resort — py.exe Windows launcher ─────────────
            if (TryPython("py")) return "py";

            return string.Empty; // not found
        }

        [System.Runtime.Versioning.SupportedOSPlatform("windows")]
        private static string? FindPythonInRegistry()
        {
            // HKCU\Software\Python\PythonCore\<version>\InstallPath
            string[] roots = {
                @"SOFTWARE\Python\PythonCore",
                @"SOFTWARE\WOW6432Node\Python\PythonCore"
            };
            RegistryHive[] hives = { RegistryHive.CurrentUser, RegistryHive.LocalMachine };

            foreach (var hive in hives)
            {
                using var baseKey = RegistryKey.OpenBaseKey(hive, RegistryView.Default);
                foreach (var root in roots)
                {
                    using var coreKey = baseKey.OpenSubKey(root);
                    if (coreKey == null) continue;

                    foreach (var version in coreKey.GetSubKeyNames())
                    {
                        using var installKey = coreKey.OpenSubKey($@"{version}\InstallPath");
                        if (installKey == null) continue;

                        string? installDir = installKey.GetValue("ExecutablePath") as string
                                          ?? installKey.GetValue("")              as string;

                        if (installDir != null)
                        {
                            string candidate = installDir.EndsWith("python.exe",
                                StringComparison.OrdinalIgnoreCase)
                                ? installDir
                                : Path.Combine(installDir, "python.exe");

                            if (File.Exists(candidate)) return candidate;
                        }
                    }
                }
            }
            return null;
        }

        private static bool TryPython(string exe)
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName               = exe,
                    Arguments              = "--version",
                    RedirectStandardOutput = true,
                    RedirectStandardError  = true,
                    UseShellExecute        = false,
                    CreateNoWindow         = true
                };
                using var p = Process.Start(psi);
                p?.WaitForExit(4000);
                return p?.ExitCode == 0;
            }
            catch { return false; }
        }

        private static ChatbotResponse FallbackResponse() => new()
        {
            Intent   = "unknown",
            Response = "I didn't quite understand that. Try asking about programming, AI, " +
                       "cybersecurity, networking, hardware, software, web development, " +
                       "databases, mobile development, or cloud computing!",
            Confidence = 0
        };
    }

    public class ChatbotResponse
    {
        public string Intent     { get; set; } = "unknown";
        public string Response   { get; set; } = string.Empty;
        public double Confidence { get; set; }
    }
}
