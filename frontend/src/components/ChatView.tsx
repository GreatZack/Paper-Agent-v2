"use client";

import {
  ActionBarPrimitive,
  AuiIf,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
} from "@assistant-ui/react";
import { ArrowUp, Check, ChevronDown, Copy, Pencil } from "lucide-react";
import { FC } from "react";
import { TooltipIconButton } from "@/components/tooltip-icon-button";
import { MarkdownText } from "@/components/markdown-text";

export const ChatView: FC = () => {
  return (
    <ThreadPrimitive.Root className="flex h-full flex-col items-stretch bg-[var(--background)] px-4 text-[var(--foreground)]">
      <AuiIf condition={(s) => s.thread.isEmpty}>
        <EmptyState />
      </AuiIf>

      <AuiIf condition={(s) => !s.thread.isEmpty}>
        <ThreadPrimitive.Viewport className="flex grow flex-col gap-8 overflow-y-scroll pt-16">
          <ThreadPrimitive.Messages>
            {({ message }) => {
              if (message.role === "user") return <UserMessage />;
              return <AssistantMessage />;
            }}
          </ThreadPrimitive.Messages>

          <ThreadPrimitive.ViewportFooter className="sticky bottom-0 mx-auto mt-auto flex w-full max-w-3xl flex-col gap-2 overflow-visible rounded-t-3xl bg-[var(--background)] pb-2">
            <ThreadScrollToBottom />
            <ChatComposer placeholder="描述一个研究课题..." />
            <p className="text-center text-xs text-[var(--muted)]">
              AI 助手可能会出错，请核实重要信息。
            </p>
          </ThreadPrimitive.ViewportFooter>
        </ThreadPrimitive.Viewport>
      </AuiIf>
    </ThreadPrimitive.Root>
  );
};

const EmptyState: FC = () => {
  return (
    <div className="flex grow flex-col items-center justify-center px-4 pb-[16vh]">
      <div className="mx-auto flex w-full max-w-3xl flex-col items-stretch gap-6">
        <h1 className="text-center text-2xl leading-7 font-normal text-[var(--foreground)]">
          你想搜索什么研究课题？
        </h1>
        <ChatComposer placeholder="例如：大语言模型安全、Transformer 注意力机制..." />
      </div>
    </div>
  );
};

const ChatComposer: FC<{ placeholder: string }> = ({ placeholder }) => {
  return (
    <ComposerPrimitive.Root className="group/composer flex w-full flex-col rounded-[28px] border border-[var(--composer-border)] bg-[var(--composer-bg)] px-2 py-2 shadow-[0_2px_6px_-2px_rgba(0,0,0,0.05)] focus-within:border-[#d0d0d0] dark:focus-within:border-transparent">
      <div className="flex items-end gap-1">
        <ComposerPrimitive.Input
          autoFocus
          placeholder={placeholder}
          rows={1}
          className="max-h-52 min-h-9 flex-1 resize-none bg-transparent py-1.5 pr-2 pl-1 text-base text-[var(--foreground)] outline-none placeholder:text-[#8e8e8e]"
        />

        <div className="flex shrink-0 items-center gap-1">
          <AuiIf condition={(s) => s.thread.isRunning}>
            <ComposerPrimitive.Cancel className="flex size-9 items-center justify-center rounded-full bg-[var(--user-bubble)] text-[var(--user-bubble-text)]">
              <div className="size-2.5 rounded-[2px] bg-current" />
            </ComposerPrimitive.Cancel>
          </AuiIf>

          <AuiIf
            condition={(s) => !s.thread.isRunning && !s.composer.isEmpty}
          >
            <ComposerPrimitive.Send className="flex size-9 items-center justify-center rounded-full bg-[var(--user-bubble)] text-[var(--user-bubble-text)] transition-opacity disabled:opacity-30">
              <ArrowUp className="size-5" />
            </ComposerPrimitive.Send>
          </AuiIf>
        </div>
      </div>
    </ComposerPrimitive.Root>
  );
};

const ThreadScrollToBottom: FC = () => {
  return (
    <ThreadPrimitive.ScrollToBottom asChild>
      <TooltipIconButton
        tooltip="滚动到底部"
        className="bg-background absolute -top-10 z-10 self-center rounded-full border p-2 shadow-sm disabled:invisible dark:border-white/15 dark:bg-[#2a2a2a]"
      >
        <ChevronDown className="size-5" />
      </TooltipIconButton>
    </ThreadPrimitive.ScrollToBottom>
  );
};

const assistantActionClassName =
  "flex size-8 items-center justify-center rounded-lg text-[var(--icon-color)] transition-colors hover:bg-[var(--icon-hover)]";

const UserMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="relative mx-auto flex w-full max-w-3xl flex-col items-end gap-1">
      <div className="max-w-[70%] rounded-[22px] bg-[var(--user-bubble)] px-4 py-2.5 leading-6 text-[var(--user-bubble-text)]">
        <MessagePrimitive.Parts />
      </div>

      <div className="flex items-center gap-0.5">
        <ActionBarPrimitive.Root
          hideWhenRunning
          autohide="always"
          autohideFloat="single-branch"
          className="flex items-center"
        >
          <ActionBarPrimitive.Copy asChild>
            <TooltipIconButton
              tooltip="复制"
              side="top"
              className={assistantActionClassName}
            >
              <AuiIf condition={(s) => s.message.isCopied}>
                <Check className="size-5" />
              </AuiIf>
              <AuiIf condition={(s) => !s.message.isCopied}>
                <Copy className="size-5" />
              </AuiIf>
            </TooltipIconButton>
          </ActionBarPrimitive.Copy>
          <ActionBarPrimitive.Edit asChild>
            <TooltipIconButton
              tooltip="编辑"
              side="top"
              className={assistantActionClassName}
            >
              <Pencil className="size-5" />
            </TooltipIconButton>
          </ActionBarPrimitive.Edit>
        </ActionBarPrimitive.Root>
      </div>
    </MessagePrimitive.Root>
  );
};

const AssistantMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="relative mx-auto flex w-full max-w-3xl flex-col">
      <div className="w-full rounded-2xl border border-[var(--assistant-card-border)] bg-[var(--assistant-bg)] px-5 py-4 text-[var(--foreground)]">
        <MessagePrimitive.Parts>
          {({ part }) => {
            if (part.type === "text") return <MarkdownText />;
            return null;
          }}
        </MessagePrimitive.Parts>
      </div>
    </MessagePrimitive.Root>
  );
};
