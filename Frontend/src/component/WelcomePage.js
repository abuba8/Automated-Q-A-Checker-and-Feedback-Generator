import React from "react"
import {PageHeader, Button, Typography, Divider, Row, Col, Card} from "antd";
import {Link} from "react-router-dom"
import '../css/Welcome.css';
import LocalStorage from 'local-storage'

const {Text} = Typography;
class WelcomePage extends React.Component {
  state = {
    size: 'large'
  }
    render(){

      const auth = LocalStorage.get('auth')
        return(
    <div className="welcomediv" style={{display: 'flex', justifyContent: 'center'}} >
      <PageHeader ghost={false} subTitle="" extra={[]}>
        <div style={{textAlign: 'center',  marginLeft: "30%", marginRight: '50%', margin: '23px'}}>
          {
            !auth ?
            <>
              <Link to="/login"><Button size="large" style={{margin: '5px'}}key="3">Login</Button></Link>
              <Link to="/registerteacher/"><Button size="large" key="2">Sign Up</Button></Link>
            </>
            :
            ''
          }

        </div>
          <p style={{textAlign: 'center', color: 'black', fontSize: '29px'}}>Here are a list of features that this web application has to offer!</p>
        
          <Row justify="space-around" align="middle">
            <Col span={8} style={{margin: '5px'}}>
              <Card className="popupwelcome"><Text strong>1. Generate multiple feedbacks for a single test!</Text></Card>
              <Card className="popupwelcome"><Text strong>2. Feedbacks generated for exams, mid terms and even for revision!</Text></Card>
            </Col>
            <Col span={8} style={{margin: '5px'}}>
              <Card className="popupwelcome"><Text strong>3. See improvement feedback for weak topics</Text></Card>
              <Card className="popupwelcome"><Text strong>4. Save and export feedback and test information</Text></Card>
            </Col>
          </Row>
      </PageHeader>  
    </div>
        )
    }
}
export default WelcomePage