import styled from 'styled-components';

export const Container = styled.div`
  width: 100%;
  display: flex;
  bottom: 0;
  flex-direction: column;
  justify-content: center;
  align-items: center;
`;

export const FooterText = styled.p`
  padding: 2px;
  margin: 0;
  color: ${(props) => props.theme.colors.onPrimary};
  ${(props) => props.theme.fonts.body.large};
  display: flex;
  align-items: center;

  @media screen and ${(props) => props.theme.device.mobileL} {
    ${(props) => props.theme.fonts.body.medium};
  } ;
`;

export const GreenText = styled(FooterText)`
  color: ${(props) => props.theme.colors.primaryContainer};
`;

export const CopyrightText = styled(FooterText)`
  ${(props) => props.theme.fonts.body.small};
`;
