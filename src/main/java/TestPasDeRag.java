import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;

public class TestPasDeRag {
    public static void main(String[] args) throws Exception {
        String cle = System.getenv("GEMINI_KEY");

        ChatModel model = GoogleAiGeminiChatModel
                .builder()
                .apiKey(cle)
                .modelName("gemini-2.0-flash-exp")
                .temperature(0.6)
                .build();
    }
}