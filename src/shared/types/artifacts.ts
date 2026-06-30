export interface ArtifactProvenance {
  workflowId?: string
  workflowNodeId?: string
  source?: string
  [key: string]: unknown
}
