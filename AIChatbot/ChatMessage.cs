using System;

namespace AIChatbot
{
    public enum MessageSender
    {
        User,
        Bot
    }

    public class ChatMessage
    {
        public string Text        { get; set; } = string.Empty;
        public MessageSender Sender { get; set; }
        public DateTime Timestamp { get; set; } = DateTime.Now;
        public string? Intent     { get; set; }
        public double Confidence  { get; set; }

        public string FormattedTime => Timestamp.ToString("hh:mm tt");
    }
}
