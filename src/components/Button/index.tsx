import { ButtonProps } from 'antd';
import React from 'react';

import { StyledButton } from './styles';

type Props = Omit<ButtonProps, 'size' | 'type'> & {
  as?: any;
  size?: any;
  to?: string;
  type:
    | 'link'
    | 'text'
    | 'default'
    | 'secondary'
    | 'ghost'
    | 'dashed'
    | 'primary'
    | undefined
    | any;
};

export default function Button(props: Props) {
  const { children, to, ...remainingProps } = props;

  return (
    <a href={to} target='_blank' rel='noreferrer'>
      <StyledButton {...remainingProps}>{children}</StyledButton>
    </a>
  );
}
