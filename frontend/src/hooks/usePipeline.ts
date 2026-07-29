"use client";

import { useCallback, useRef, useState } from "react";
import { PipelineSocket, PipelineStatus } from "@/lib/websocket";

export interface PipelineState {
  status: PipelineStatus;
  currentStep: string;
  progressMessage: string;
  report: string | null;
  error: string | null;
  startPipeline: (query: string, maxPapers?: number) => Promise<string>;
  reset: () => void;
}

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/pipeline";

export function usePipeline(): PipelineState {
  const [status, setStatus] = useState<PipelineStatus>("idle");
  const [currentStep, setCurrentStep] = useState("");
  const [progressMessage, setProgressMessage] = useState("");
  const [report, setReport] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const socketRef = useRef<PipelineSocket | null>(null);

  const startPipeline = useCallback((query: string, maxPapers?: number) => {
    return new Promise<string>((resolve, reject) => {
      setError(null);
      setReport(null);
      setCurrentStep("");
      setProgressMessage("");

      if (socketRef.current) {
        socketRef.current.close();
      }

      const socket = new PipelineSocket(WS_URL, {
        onProgress: (step, state, data) => {
          if (step === "heartbeat") {
            setProgressMessage(
              typeof data === "string" ? data : "任务仍在处理中",
            );
            return;
          }
          setCurrentStep(step);
          setProgressMessage(typeof data === "string" ? data : `${step}...`);
        },
        onReport: (reportText) => {
          setReport(reportText);
          setStatus("completed");
          resolve(reportText);
        },
        onError: (err) => {
          setError(err);
          setStatus("failed");
          reject(new Error(err));
        },
        onStatusChange: (s) => {
          setStatus(s);
        },
      });

      socketRef.current = socket;
      socket.connect(query, maxPapers);
    });
  }, []);

  const reset = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.close();
    }
    setStatus("idle");
    setCurrentStep("");
    setProgressMessage("");
    setReport(null);
    setError(null);
  }, []);

  return {
    status,
    currentStep,
    progressMessage,
    report,
    error,
    startPipeline,
    reset,
  };
}
