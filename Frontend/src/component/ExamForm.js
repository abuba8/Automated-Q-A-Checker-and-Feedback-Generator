import { Form, Input, Button, Select} from 'antd';
import React from "react"
import axios from "axios";
import { push } from 'react-router-redux';
import {Link} from "react-router-dom";
const {Option} = Select;

class ExamForm extends React.Component {
  constructor(props){
    super(props);
    this.state = {
      type: 'default',
      module: [],
      module_id: 0,
      showAlert: false,
    }
    this.handleChange = this.handleChange.bind(this);
  }
  handleChange = (e) => {
    const val = e.label
    console.log(val)
    this.setState({
      type: e.label
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
    }
        //request type given because form is used more than once
        //also specifying the module id for updating a specific module.
  handleFormSubmit = (e, requestMethod, moduleID, questionID) => {
      e.preventDefault();
      const title = e.target.elements.name.value;
      const description = e.target.elements.description.value;
      const num_questions = e.target.elements.num.value;
      const num_sub = e.target.elements.sub.value;
      const type = this.state.type;
      console.log(questionID)
      console.log(title, num_questions, description, type);

      switch(requestMethod) {
        case 'post':
          axios.post(`http://127.0.0.1:8000/api/create/test/`, {
            name: title,
            test_count: num_questions,
            num_subquestions: num_sub,
            description: description,
            created_date: new Date().toLocaleString(),
            questiontype: type,
            module: moduleID,
          })
          .then(res => {
            console.log(res) 
              //redirect to home page after creating
              this.setState({
                showingAlert: true,
                module_id: moduleID
              });
              setTimeout(() => {
                this.setState({
                  showingAlert: false,
                });
              }, 5000);
          })
          .catch(err => console.log(err))
        case 'put':
            axios.put(`http://127.0.0.1:8000/api/test/${questionID}/update/`, {
              name: title,
              test_count: num_questions,
              description: description,
              created_date: new Date().toLocaleString(),
              questiontype: type,
              module: moduleID,
            })
            .then(res => {
              console.log(res)
              alert("the test you have created has been saved")
            })
            .catch(err => console.log(err))
        default:
          return null
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
                <strong>Your test has been saved!</strong><Link to={'/feedbackstage/' + this.state.module_id + '/' + this.props.userID}><Button type="primary" htmlType="submit" style={{alignItems:'center'}}>Click to check your test and grade!</Button></Link>
          </div>
          <Form onSubmit={(event) => this.handleFormSubmit(event, this.props.requestMethod, this.props.moduleID, this.props.questionID)}>
              <Form.Item label="Name">
                <Input name="name" placeholder="Name of exam..." />
              </Form.Item>
              <Form.Item label="Number of questions">
                <Input type="number" name="num" pattern="[0-9]*" onKeyPress={this.onKeyPress.bind(this)} />
              </Form.Item>
              <Form.Item label="Number of sub questions">
                <Input type="number" name="sub" pattern="[0-9]*" onKeyPress={this.onKeyPress.bind(this)} />
              </Form.Item>
              <Form.Item label="Description">
                <Input name="description" name="description" placeholder="e.g. mid term or revision..." />
              </Form.Item>
              <Form.Item label="Type of question">
                  <Select ref={ref => {
                    this._select = ref }} labelInValue defaultValue={this.state.value} style={{width: 120}} onChange={this.handleChange}>
                      <Option value="MCQ">Multiple choice questions</Option>
                      <Option value="Definition">Short definitions</Option>
                      <Option value="Skeleton Program">Skeleton program questions</Option>
                </Select>
              </Form.Item>
              <Form.Item>
                <Button type="primary" htmlType="submit">{this.props.btnText}</Button>
              </Form.Item>
        </Form>
      </div>
    );
  }
}
export default ExamForm