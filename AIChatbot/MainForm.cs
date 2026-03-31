using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace AIChatbot
{
    public partial class MainForm : Form
    {
        private ChatbotEngine _engine;
        private FlowLayoutPanel flpChat;
        private Panel chatWrapper;
        private TextBox txtInput;
        private Button btnSend;
        private Panel typingPanel;
        private bool _isProcessing = false;

        // 🎨 Colors (ChatGPT style)
        Color bg = Color.FromArgb(11, 11, 15);
        Color botBubble = Color.FromArgb(35, 35, 45);
        Color userBubble = Color.FromArgb(0, 120, 215);
        Color inputBg = Color.FromArgb(25, 25, 35);
        Color textColor = Color.White;
        Color accent = Color.FromArgb(0, 120, 215);

        public MainForm()
        {
            _engine = new ChatbotEngine();
            BuildUI();
            _ = Init();
        }

        // 🧱 UI
        private void BuildUI()
        {
            Text = "TechBot – AI Assistant";
            Size = new Size(1100, 750);
            BackColor = bg;

            // Chat wrapper (centered)
            chatWrapper = new Panel
            {
                Dock = DockStyle.Fill,
                Padding = new Padding(160, 10, 160, 10)
            };

            flpChat = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                FlowDirection = FlowDirection.TopDown,
                WrapContents = false,
                AutoScroll = true
            };

            chatWrapper.Controls.Add(flpChat);
            Controls.Add(chatWrapper);

            // Typing indicator
            typingPanel = new Panel
            {
                Height = 30,
                Dock = DockStyle.Bottom,
                Visible = false
            };

            typingPanel.Controls.Add(new Label
            {
                Text = "🤖 typing...",
                ForeColor = Color.Gray,
                Dock = DockStyle.Fill
            });

            Controls.Add(typingPanel);

            // Input
            var bottom = new Panel
            {
                Dock = DockStyle.Bottom,
                Height = 70,
                Padding = new Padding(160, 10, 160, 10)
            };

            var inputBox = new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = inputBg
            };

            txtInput = new TextBox
            {
                Dock = DockStyle.Fill,
                BorderStyle = BorderStyle.None,
                BackColor = inputBg,
                ForeColor = textColor,
                Font = new Font("Segoe UI", 11),
                Padding = new Padding(10)
            };

            btnSend = new Button
            {
                Text = "➤",
                Dock = DockStyle.Right,
                Width = 60,
                BackColor = accent,
                ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat
            };
            btnSend.FlatAppearance.BorderSize = 0;
            btnSend.Click += async (s, e) => await Send();

            txtInput.KeyDown += async (s, e) =>
            {
                if (e.KeyCode == Keys.Enter)
                {
                    e.SuppressKeyPress = true;
                    await Send();
                }
            };

            inputBox.Controls.Add(txtInput);
            inputBox.Controls.Add(btnSend);

            inputBox.Resize += (s, e) => Round(inputBox, 20);

            bottom.Controls.Add(inputBox);
            Controls.Add(bottom);
        }

        private async Task Init()
        {
            var (ok, msg) = await _engine.InitialiseAsync();

            if (ok)
                await AddBotMessage("👋 Hello! I'm your AI assistant.");
            else
                await AddBotMessage(msg);
        }

        // 📩 SEND
        private async Task Send()
        {
            if (_isProcessing) return;

            string text = txtInput.Text.Trim();
            if (text == "") return;

            _isProcessing = true;
            txtInput.Clear();

            AddUserMessage(text);

            typingPanel.Visible = true;

            var res = await _engine.PredictAsync(text);

            typingPanel.Visible = false;

            await AddBotMessage(res.Response);

            _isProcessing = false;
        }

        // 👤 USER MSG
        private void AddUserMessage(string text)
        {
            AddMessage(text, true);
        }

        // 🤖 BOT MSG (STREAMING)
        private async Task AddBotMessage(string text)
        {
            int maxWidth = (int)(flpChat.ClientSize.Width * 0.6);

            Label lbl = new Label
            {
                ForeColor = textColor,
                BackColor = botBubble,
                Font = new Font("Segoe UI", 10),
                MaximumSize = new Size(maxWidth, 0),
                AutoSize = true,
                Padding = new Padding(12)
            };

            Panel bubble = new Panel
            {
                AutoSize = true,
                BackColor = botBubble
            };

            bubble.Controls.Add(lbl);
            bubble.Resize += (s, e) => Round(bubble, 15);

            Button copyBtn = new Button
            {
                Text = "📋",
                Width = 30,
                Height = 25,
                FlatStyle = FlatStyle.Flat,
                BackColor = botBubble,
                ForeColor = Color.White
            };
            copyBtn.FlatAppearance.BorderSize = 0;
            copyBtn.Click += (s, e) => Clipboard.SetText(text);

            Panel row = new Panel
            {
                Width = flpChat.ClientSize.Width - 20,
                Height = bubble.Height + 10
            };

            bubble.Location = new Point(5, 0);
            copyBtn.Location = new Point(bubble.Right + 5, 5);

            row.Controls.Add(bubble);
            row.Controls.Add(copyBtn);

            flpChat.Controls.Add(row);

            // ✨ Streaming effect
            string current = "";
            foreach (char c in text)
            {
                current += c;
                lbl.Text = current;
                await Task.Delay(10);
                flpChat.ScrollControlIntoView(row);
            }
        }

        // 💬 COMMON MESSAGE
        private void AddMessage(string text, bool isUser)
        {
            int maxWidth = (int)(flpChat.ClientSize.Width * 0.6);

            Label lbl = new Label
            {
                Text = text,
                ForeColor = textColor,
                BackColor = isUser ? userBubble : botBubble,
                Font = new Font("Segoe UI", 10),
                MaximumSize = new Size(maxWidth, 0),
                AutoSize = true,
                Padding = new Padding(12)
            };

            Panel bubble = new Panel
            {
                AutoSize = true,
                BackColor = lbl.BackColor
            };

            bubble.Controls.Add(lbl);
            bubble.Resize += (s, e) => Round(bubble, 15);

            Panel container = new Panel
            {
                Width = flpChat.ClientSize.Width - 20,
                Height = bubble.Height + 10
            };

            if (isUser)
                bubble.Location = new Point(container.Width - bubble.Width - 5, 0);
            else
                bubble.Location = new Point(5, 0);

            container.Controls.Add(bubble);
            flpChat.Controls.Add(container);

            flpChat.ScrollControlIntoView(container);
        }

        // 🎨 ROUND CORNERS
        private void Round(Control c, int r)
        {
            GraphicsPath path = new GraphicsPath();
            path.AddArc(0, 0, r, r, 180, 90);
            path.AddArc(c.Width - r, 0, r, r, 270, 90);
            path.AddArc(c.Width - r, c.Height - r, r, r, 0, 90);
            path.AddArc(0, c.Height - r, r, r, 90, 90);
            path.CloseAllFigures();
            c.Region = new Region(path);
        }
    }
}