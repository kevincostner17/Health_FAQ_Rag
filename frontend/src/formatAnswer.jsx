import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** Turn 【Source: Topic | Source】 into readable citation chips in plain text first. */
function normalizeCitations(text) {
  return text.replace(/【Source:\s*([^|】]+)\s*\|\s*([^】]+)】/g, " *(Source: $1 · $2)*");
}

export default function FormattedAnswer({ text, refused }) {
  if (!text) return null;

  const content = normalizeCitations(text);

  return (
    <div className={`answer-body ${refused ? "answer-refused" : ""}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p className="answer-p">{children}</p>,
          strong: ({ children }) => <strong className="answer-strong">{children}</strong>,
          ul: ({ children }) => <ul className="answer-ul">{children}</ul>,
          ol: ({ children }) => <ol className="answer-ol">{children}</ol>,
          li: ({ children }) => <li className="answer-li">{children}</li>,
          em: ({ children }) => <em className="answer-cite">{children}</em>,
          h1: ({ children }) => <h3 className="answer-h">{children}</h3>,
          h2: ({ children }) => <h3 className="answer-h">{children}</h3>,
          h3: ({ children }) => <h3 className="answer-h">{children}</h3>,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
