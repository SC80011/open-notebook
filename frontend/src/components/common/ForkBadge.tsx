'use client'

import { useTranslation } from '@/lib/hooks/use-translation'
import { cn } from '@/lib/utils'

interface ForkBadgeProps {
  className?: string
}

/** Visual-only build marker ("cc" = concurrent). Does not affect application behavior. */
export function ForkBadge({ className }: ForkBadgeProps) {
  const { t } = useTranslation()
  return (
    <span
      className={cn(
        'ml-1 rounded px-1 font-mono text-[0.65rem] font-semibold uppercase tracking-wide text-sky-700 dark:text-sky-300',
        className
      )}
      aria-hidden
    >
      {t('common.forkBadge')}
    </span>
  )
}
