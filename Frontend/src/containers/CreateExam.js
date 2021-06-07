import React from "react";
import ExamForm from "../component/ExamForm";
import axios from "axios"
class CreateExam extends React.Component{
    constructor(props){
        super(props);
        this.state = {
          num: 0,
          module: []
        }
      }
    componentDidMount () {
        axios.get(`http://127.0.0.1:8000/api/modules`)
        .then(res => {
            this.setState({
                module: res.data
            })
            console.log(res.data)
        })
    }
    render()
    {
        let title = ""
        let id = this.props.match.params.id;
        let tempmodules = this.state.module;
        tempmodules.map(function(item, i){
            if(item.id == id)
            {
                title = item.title;
            }
        })
        return(
            <div>
                
                    <h2 style={{display: 'flex', justifyContent: 'center'}} >Enter Exam information here for {title}!</h2>
                    {console.log(this.props.match.params.id)}
                    <ExamForm requestMethod="post"
                        moduleID={this.props.match.params.id}
                        userID={this.props.match.params.userid}
                        questionID={null}
                        btnText="Create an exam!" />
            </div>
        )
    }
}
export default CreateExam