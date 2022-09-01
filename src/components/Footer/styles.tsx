import styled from 'styled-components';

export const Container = styled.div`
  width: 100%;
  height: 30px;
  display: flex;
  padding: 40px 0 30px 0;
  bottom: 0;
  flex-direction: column;
  justify-content: center;
  align-items: center;
`;

export const FooterText = styled.p`
  padding: 2px;
  margin: 0;
  color: ${(props) => props.theme.colors.mediumGray};
  font-size: ${(props) => props.theme.fontSizes.small};
  display: flex;
  flex-direction: row;
  align-items: center;

  @media screen and ${(props) => props.theme.device.mobileL} {
    font-size: ${(props) => props.theme.fontSizes.normal};
  } ;
`;

export const GreenText = styled(FooterText)`
  color: ${(props) => props.theme.colors.lightGreen};
`;

export const CopyrightText = styled(FooterText)`
  font-size: ${(props) => props.theme.fontSizes.small};
`;
