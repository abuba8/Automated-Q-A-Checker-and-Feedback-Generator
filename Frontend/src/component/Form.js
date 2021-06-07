import { Form, Input, Button, Modal} from 'antd';
import React from "react"
import axios from "axios";
import {Link, Redirect} from "react-router-dom";
class CustomForm extends React.Component {
  constructor(props){
    super(props);
    this.state = {
      num: 0,
      module: [],
      showingAlert: false
    }
  }
  handleChange = (e) => {
    this.setState({
      num: e,
    })
  }
  componentDidMount()
    {
        const moduleID = this.props.moduleID;
        axios.get(`http://127.0.0.1:8000/api/module/${moduleID}`)
        .then(res => {
            this.setState({
                module: res.data
            })
            console.log(res.data)
        })
        console.log(this.props.userID)
    }
        //request type given because form is used more than once
        //also specifying the module id for updating a specific module.
  handleFormSubmit = (e, requestMethod, moduleID, userID) => {
      e.preventDefault();
      const title = e.target.elements.title.value;
      const code = e.target.elements.code.value;
      const year = e.target.elements.year.value;
      const description = e.target.elements.description.value;
      const num_students = e.target.elements.num.value;
      
      console.log(title, description, code, num_students);
      
      switch(requestMethod) {
        case 'post':
          let user_data = []
          let found = false;
          axios.get(`http://127.0.0.1:8000/api/users`)
          .then(res => {
            user_data = res.data
            user_data.map(function(item, i){
                if(item.username == userID){
                    console.log(item.username)
                    axios.post(`http://127.0.0.1:8000/api/create/module/`, {
                      user: item.id,
                      title: title,
                      year_of_module: year,
                      code: code,
                      description: description,
                      num_students: num_students,
                    })
                    .then(res => {
                        console.log(res)
                        found = true;
                        let secondsToGo = 60;
                        const modal = Modal.success({
                          title: `This message will be destroyed after 1 minute.`,
                          content: ``,
                        });
                        const timer = setInterval(() => {
                          secondsToGo -= 1;
                          modal.update({
                            content: <div>
                                        <strong>You have created a new module, go and check it out under your modules list!</strong>
                                      </div>,
                          });
                        }, 1000);
                        setTimeout(() => {
                          clearInterval(timer);
                          modal.destroy();
                        }, secondsToGo * 1000);
                    })
                    .catch(err => {
                        console.log("Error")
                    })
                }
            })
            if(found)
            {
              this.setState({
                showingAlert: true
              });
              setTimeout(() => {
                this.setState({
                  showingAlert: false,
                });
              }, 5000);
            }
            else
            {
              
            }
            console.log(user_data)
        })
        case 'put':
            axios.put(`http://127.0.0.1:8000/api/module/${moduleID}/update/`, {
              title: title,
              year: year,
              code: code,
              description: description,
              num_students: num_students,
            })
            .then(res => console.log(res))
            .catch(err => console.log(err))
        default:
          return "Couldnt add the module!"
      }
  }
  onKeyPress(event) {
    const keyCode = event.keyCode || event.which;
    const keyValue = String.fromCharCode(keyCode);
     if (/\+|-/.test(keyValue))
       event.preventDefault();
  }
  render() {
    return (
      <div>
        <div className={`alert alert-success ${this.state.showingAlert ? 'alert-shown': 'alert-hidden'}`}>
                <strong>You have created a new module</strong><Link to={'/modulesList/' + this.props.userID + '/'}><Button type="primary" htmlType="submit" style={{alignItems:'center'}}>Click to check your saved modules!</Button></Link>
          </div>
          <Form onSubmit={(event) => this.handleFormSubmit(event, this.props.requestMethod, this.props.moduleID, this.props.userID)}>
              <Form.Item label="Title">
                <Input name="title" placeholder={this.state.module.title} />
              </Form.Item>
              <Form.Item label="Year">
                <Input type="number" name="year" pattern="[0-4]*" onKeyPress={this.onKeyPress.bind(this)} />
              </Form.Item>
              <Form.Item label="Code">
                <Input name="code" placeholder={this.state.module.code} />
              </Form.Item>
              <Form.Item label="Description">
                <Input name="description" placeholder={this.state.module.description} />
              </Form.Item>
              <Form.Item label="Number Of Students">
                <Input type="number" name="num" pattern="[0-400]" onKeyPress={this.onKeyPress.bind(this)} placeholder={this.state.module.num_students} />
              </Form.Item>
          
          <Form.Item>
            <Button type="primary" htmlType="submit">{this.props.btnText}</Button>
          </Form.Item>
        </Form>
      </div>
    );
  }
}
export default CustomForm