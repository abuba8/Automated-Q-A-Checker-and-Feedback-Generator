import { Form, Input, Button, Select} from 'antd';
import React from "react"
import axios from "axios";
const {TextArea} = Input;
const {Option} = Select;

class FeedbackForm extends React.Component {
  constructor(props){
    super(props);
    this.state = {
      test_id : 0,
      grade_id : 0,
      showingAlert: false,
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
            let id = this.props.testID
            console.log(id)
            let test_id = 0
            let grade_id = 0
            axios.all([
                axios.get('http://127.0.0.1:8000/api/test'),
                axios.get('http://127.0.0.1:8000/api/grades'),
                
            ])  //look through both the test and grade tables before saving into savedfeedback model
            .then(axios.spread((testres, grades) => {
                    testres.data.map(function(item, i){
                        if(item.name == id)
                        {
                            console.log(item.id)
                            test_id = item.id
                        }
                    })
                    grades.data.map(function(grade, i){
                        if(grade.test == test_id)
                        {
                            grade_id = grade.id
                        }
                    })
                    this.setState({
                        test_id: test_id,
                        grade_id: grade_id
                    })
            }))
    }
        //request type given because form is used more than once
        //also specifying the module id for updating a specific module.
  handleFormSubmit = (e, requestMethod, testID, userID) => {
      e.preventDefault();
      const feedback = e.target.elements.feedback.value;
      console.log(feedback);
      switch(requestMethod) {
        case 'post':
            console.log(this.state.test_id + " " + this.state.grade_id + " " + userID)
            axios.post(`http://127.0.0.1:8000/api/save/feedback/`,{
                    test: this.state.test_id,
                    grade: this.state.grade_id,
                    user: userID,
                    feedback: feedback,
                    percentage: 100,
                    created_by: userID
            })
            this.setState({
              showingAlert: true
          });
          setTimeout(() => {
              this.setState({
                showingAlert: false,
              });
          }, 5000);
            console.log("r")
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
                <strong>Your Individual feedback has been saved!</strong> 
        </div>
          <Form onSubmit={(event) => this.handleFormSubmit(event, this.props.requestMethod, this.props.testID, this.props.userID)}>
              <Form.Item label="Feedback">
                <TextArea name="feedback" rows={4} placeholder="Enter feedback..." />
              </Form.Item>
              <Form.Item>
                <Button type="primary" htmlType="submit">{this.props.btnText}</Button>
              </Form.Item>
        </Form>
      </div>
    );
  }
}
export default FeedbackForm