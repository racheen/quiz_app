// components/DynamicText.tsx
import React from 'react';
import { InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

type Props = {
  text: string;
};

const DynamicText: React.FC<Props> = ({ text }) => {
  const parts = text.split(/(\$[^$]+\$)/g); // Split by $...$

  return (
    <>
      {parts.map((part, index) =>
        part.startsWith('$') && part.endsWith('$') ? (
          <InlineMath key={index} math={part.slice(1, -1)} />
        ) : (
          <span key={index}>{part}</span>
        )
      )}
    </>
  );
};

export default DynamicText;
