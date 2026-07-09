"use client";

import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { ChatView } from "@/components/ChatView";
import { usePipelineRuntime } from "@/hooks/usePipelineRuntime";

export default function Home() {
  const { runtime } = usePipelineRuntime();

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <main className="h-dvh">
        <ChatView />
      </main>
    </AssistantRuntimeProvider>
  );
}
