import React from "react";
import { Form, Input, Button } from 'antd';
import "../css/Layout.css";
const layout = {
  labelCol: { span: 8 },
  wrapperCol: { span: 16 },
};
const validateMessages = {
    required: '${label} is required!',
    types: {
      email: '${label} is not validate email!',
      number: '${label} is not a validate number!',
    },
    number: {
      range: '${label} must be between ${min} and ${max}',
    },
  };
class ContactUs extends React.Component {
  constructor(props)
  {
      super(props)
      this.state = {
          name: "",
          contactSubmit: false
      }
  }
  handleSubmit(e){
    const name = e.target.elements.name.value;
    console.log(name)
    this.setState({
      name: name,
      contactSubmit: true
    })

  }
  render(){
        const onFinish = values => {
            console.log(values);
            this.state.contactSubmit = true
        };
        return(
            <div className="contact" style={{textAlign: 'center'}}>
              {this.state.contactSubmit 
              ? 
              <div style={{color: 'skyblue', fontSize: '29px'}}>
                Thank you {this.state.name}!
                <br /> 
                your message has been sent to the team. We will get back to you very shortly.
                <br />
                In the mean time please check the FAQ page to see if your query already exists.
              </div>
              :
              <Form {...layout} name="nest-messages"onSubmit={(event) => this.handleSubmit(event)} onFinish={onFinish} validateMessages={validateMessages} className="login-form">
                  <Form.Item name={['user', 'name']} label="Name" rules={[{ required: true }]}>
                      <Input name="name" />
                  </Form.Item>
                  <Form.Item name={['user', 'email']} label="Email" rules={[{ type: 'email' }]}>
                      <Input name="email" />
                  </Form.Item>
                  <Form.Item name={['user', 'introduction']} label="Message">
                      <Input.TextArea name="message" />
                  </Form.Item>
                  <Form.Item wrapperCol={{ ...layout.wrapperCol, offset: 8 }}>
                    <Button type="primary" htmlType="submit">
                        Submit
                    </Button>
                  </Form.Item>
              </Form>
              }
              
            </div>
        )
    }
}

export default ContactUs