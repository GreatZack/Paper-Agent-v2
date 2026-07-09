"use client";

import { useCallback, useState } from "react";
import { useExternalStoreRuntime, type AppendMessage, type ThreadMessage } from "@assistant-ui/react";
import { usePipeline } from "@/hooks/usePipeline";

function createUserMessage(text: string): ThreadMessage {
  return {
    id: crypto.randomUUID(),
    role: "user",
    content: [{ type: "text" as const, text }],
    attachments: [],
    createdAt: new Date(),
    metadata: { custom: {} },
  } as unknown as ThreadMessage;
}

function createAssistantMessage(text: string, status?: ThreadMessage["status"]): ThreadMessage {
  return {
    id: crypto.randomUUID(),
    role: "assistant",
    content: [{ type: "text" as const, text }],
    status: status ?? { type: "complete", reason: "stop" },
    createdAt: new Date(),
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {},
    },
  } as unknown as ThreadMessage;
}

export function usePipelineRuntime() {
  const [messages, setMessages] = useState<ThreadMessage[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const pipeline = usePipeline();

  const onNew = useCallback(async (message: AppendMessage) => {
    const textPart = message.content.find((p) => p.type === "text");
    if (!textPart || textPart.type !== "text") return;

    setMessages((prev) => [...prev, createUserMessage(textPart.text)]);
    setIsRunning(true);

    try {
      const report = await pipeline.startPipeline(textPart.text);
      setMessages((prev) => [...prev, createAssistantMessage(report)]);
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : "Pipeline failed";
      setMessages((prev) => [
        ...prev,
        createAssistantMessage(`Error: ${errMsg}`, {
          type: "incomplete",
          reason: "error",
          error: errMsg,
        }),
      ]);
    } finally {
      setIsRunning(false);
    }
  }, [pipeline]);

  const runtime = useExternalStoreRuntime({
    messages,
    isRunning,
    onNew,
  });

  return { runtime };
}
